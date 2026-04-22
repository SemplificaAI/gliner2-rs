import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

def main():
    api = HfApi()
    repo_id = "SemplificaAI/gliner2-multi-v1-onnx"
    
    # Path calculations based on the script location
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    models_dir = base_dir / "rust_component" / "models" / "gliner2-multi-v1-onnx-v2"
    readme_path = base_dir / "README_HF.md"

    if not models_dir.exists():
        print(f"Errore: la cartella {models_dir} non esiste. Assicurati di aver eseguito prima l'esportazione V2.")
        sys.exit(1)
        
    if not readme_path.exists():
        print(f"Errore: il file {readme_path} non esiste.")
        sys.exit(1)

    print(f"Inizio l'upload dei modelli V2 sul repository: {repo_id}")
    print("=" * 60)

    # 1. Carica i file fp16_v2
    print("Caricamento della cartella 'fp16_v2/' (Modelli V2 IOBinding FP16)...")
    fp16_files = [f for f in models_dir.iterdir() if f.name.endswith("_fp16.onnx") or f.name.endswith("_fp16_iobinding.onnx") or f.name == "tokenizer.json"]
    for f in fp16_files:
        path_in_repo = f"fp16_v2/{f.name}"
        print(f"  -> Uploading {f.name} in {path_in_repo}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )

    print("-" * 60)

    # 2. Carica i file fp32_v2
    print("Caricamento della cartella 'fp32_v2/' (Modelli V2 Fusi FP32)...")
    fp32_files = [f for f in models_dir.iterdir() if f.name.endswith("_fp32.onnx") or f.name == "tokenizer.json"]
    for f in fp32_files:
        path_in_repo = f"fp32_v2/{f.name}"
        print(f"  -> Uploading {f.name} in {path_in_repo}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )

    print("-" * 60)

    # 3. Carica il nuovo README.md
    print("Caricamento del nuovo README.md...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )

    print("=" * 60)
    print("✅ Upload completato con successo su Hugging Face!")
    print(f"URL: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
