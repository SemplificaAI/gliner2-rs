"""
End-to-end equivalence check: Rust/ONNX against the PyTorch reference.

Runs the PyTorch checkpoint over `tests/cases.json`, then diffs the result
against the JSON produced by the `dump_json` example. Per-fragment parity
(`verify_parity.py`) proves the graphs are faithful; this proves the whole
pipeline is — prompt construction, routing, span decoding and NMS included,
none of which live inside the ONNX graphs.

Usage:
    # 1. reference
    python compare_with_pytorch.py reference \\
        --model_path fastino/gliner2-privacy-filter-PII-multi \\
        --cases ../tests/cases.json --out /tmp/pytorch.json

    # 2. candidate
    ORT_DYLIB_PATH=... cargo run --release --example dump_json -- \\
        models/pii-onnx tests/cases.json > /tmp/rust.json

    # 3. diff
    python compare_with_pytorch.py diff --reference /tmp/pytorch.json --candidate /tmp/rust.json

Exits 1 if any case differs beyond tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# A span matches when text, label and character offsets agree exactly; the
# score may differ by this much, which is FP noise plus the FP16 gap.
SCORE_TOL = 0.02


def build_reference(model_path: str, cases_path: Path, out_path: Path) -> None:
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_path)
    cases = json.loads(cases_path.read_text())

    results = []
    for case in cases:
        out = model.extract_entities(
            case["text"], case["entities"], threshold=0.5,
            include_confidence=True, include_spans=True,
        )
        rows = []
        for label, spans in (out.get("entities") or {}).items():
            for sp in spans:
                rows.append({
                    "text": sp["text"],
                    "label": label,
                    "start": int(sp["start"]),
                    "end": int(sp["end"]),
                    "score": round(float(sp.get("confidence", 1.0)), 4),
                })
        rows.sort(key=lambda r: (r["start"], r["end"], r["label"]))
        results.append({"name": case["name"], "entities": rows})

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    total = sum(len(r["entities"]) for r in results)
    print(f"reference written to {out_path}: {len(results)} cases, {total} spans")


def build_moderation_reference(model_path: str, suite_path: Path, out_path: Path) -> None:
    """PyTorch reference for the moderation suite (safety, jailbreak, toxicity)."""
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_path)
    suite = json.loads(suite_path.read_text())

    schema = {}
    for name, spec in suite["tasks"].items():
        schema[name] = (
            {"labels": spec["labels"], "multi_label": True,
             "cls_threshold": spec["threshold"]}
            if spec["multi_label"] else spec["labels"]
        )

    results = []
    for case in suite["cases"]:
        res = model.classify_text(case["text"], schema, threshold=0.5)
        row = {}
        for name, spec in suite["tasks"].items():
            v = res.get(name)
            row[name] = sorted(v) if isinstance(v, list) else v
        results.append({"name": case["name"], "result": row})

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"moderation reference written to {out_path}: {len(results)} cases")


def diff_moderation(reference_path: Path, candidate_path: Path) -> int:
    ref = {r["name"]: r["result"] for r in json.loads(reference_path.read_text())}
    got = {r["name"]: r["result"] for r in json.loads(candidate_path.read_text())}
    failures = 0
    for name in sorted(set(ref) | set(got)):
        r, g = ref.get(name, {}), got.get(name, {})
        for task in sorted(set(r) | set(g)):
            rv, gv = r.get(task), g.get(task)
            ok = rv == gv
            failures += 0 if ok else 1
            print(f"{name:<32} {task:<22} {'OK' if ok else 'DIFF'}")
            if not ok:
                print(f"    PyTorch: {rv}")
                print(f"    ONNX   : {gv}")
    print()
    print("FAILED" if failures else "all moderation cases match")
    return 1 if failures else 0


def _key(row: dict) -> tuple:
    return (row["start"], row["end"], row["label"])


def diff(reference_path: Path, candidate_path: Path) -> int:
    ref = {r["name"]: r["entities"] for r in json.loads(reference_path.read_text())}
    got = {r["name"]: r["entities"] for r in json.loads(candidate_path.read_text())}

    names = sorted(set(ref) | set(got))
    failures = 0
    worst_score = 0.0
    total_ref = total_common = 0

    print(f"{'case':<20} {'ref':>4} {'onnx':>5} {'shared':>7} {'max dscore':>11}  result")
    print("-" * 62)

    for name in names:
        r = ref.get(name, [])
        g = got.get(name, [])
        rmap = {_key(x): x for x in r}
        gmap = {_key(x): x for x in g}
        common = set(rmap) & set(gmap)
        missing = set(rmap) - set(gmap)
        extra = set(gmap) - set(rmap)

        dmax = max((abs(rmap[k]["score"] - gmap[k]["score"]) for k in common), default=0.0)
        worst_score = max(worst_score, dmax)
        total_ref += len(r)
        total_common += len(common)

        ok = not missing and not extra and dmax <= SCORE_TOL
        failures += 0 if ok else 1
        print(f"{name:<20} {len(r):>4} {len(g):>5} {len(common):>7} {dmax:>11.4f}  "
              f"{'OK' if ok else 'DIFF'}")

        for k in sorted(missing):
            print(f"    only in PyTorch : {rmap[k]['label']:<16} {rmap[k]['text']!r}")
        for k in sorted(extra):
            print(f"    only in ONNX    : {gmap[k]['label']:<16} {gmap[k]['text']!r}")
        for k in sorted(common):
            d = abs(rmap[k]["score"] - gmap[k]["score"])
            if d > SCORE_TOL:
                print(f"    score {rmap[k]['label']:<14} {rmap[k]['text']!r}: "
                      f"{rmap[k]['score']:.4f} vs {gmap[k]['score']:.4f}")

    print()
    recall = total_common / total_ref if total_ref else 1.0
    print(f"spans: {total_common}/{total_ref} identical ({recall:.1%}), "
          f"max score delta {worst_score:.4f}")
    print("FAILED" if failures else "all cases match")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ref = sub.add_parser("reference", help="run the PyTorch checkpoint")
    ref.add_argument("--model_path", required=True)
    ref.add_argument("--cases", required=True)
    ref.add_argument("--out", required=True)

    d = sub.add_parser("diff", help="compare two result files")
    d.add_argument("--reference", required=True)
    d.add_argument("--candidate", required=True)

    mr = sub.add_parser("moderation", help="run the PyTorch moderation reference")
    mr.add_argument("--model_path", required=True)
    mr.add_argument("--cases", required=True)
    mr.add_argument("--out", required=True)

    md = sub.add_parser("diff-moderation", help="compare two moderation result files")
    md.add_argument("--reference", required=True)
    md.add_argument("--candidate", required=True)

    args = p.parse_args()
    if args.cmd == "reference":
        build_reference(args.model_path, Path(args.cases), Path(args.out))
        return 0
    if args.cmd == "moderation":
        build_moderation_reference(args.model_path, Path(args.cases), Path(args.out))
        return 0
    if args.cmd == "diff-moderation":
        return diff_moderation(Path(args.reference), Path(args.candidate))
    return diff(Path(args.reference), Path(args.candidate))


if __name__ == "__main__":
    sys.exit(main())
