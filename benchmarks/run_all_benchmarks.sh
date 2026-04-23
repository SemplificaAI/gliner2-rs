#!/bin/bash


echo "=== BENCHMARK PYTORCH NVIDIA RTX 4090 ==="
CUDA_VISIBLE_DEVICES=0 /mnt/crucial/jugaad/experiments/edito-gliner2/venv/bin/python benchmark_python.py --device cuda

echo ""
echo "=== BENCHMARK PYTORCH NVIDIA RTX 3090 ==="
CUDA_VISIBLE_DEVICES=1 /mnt/crucial/jugaad/experiments/edito-gliner2/venv/bin/python benchmark_python.py --device cuda

echo ""
echo "=== BENCHMARK PYTORCH CPU AMD RYZEN ==="
CUDA_VISIBLE_DEVICES="" /mnt/crucial/jugaad/experiments/edito-gliner2/venv/bin/python benchmark_python.py --device cpu

echo ""
echo "=== COMPILE RUST BENCHMARKS ==="
cd rust_component
cargo build --release --examples

echo ""
echo "=== BENCHMARK RUST V1 NVIDIA RTX 4090 FP32 ==="
cd ..
CUDA_VISIBLE_DEVICES=0 ./rust_component/target/release/examples/test_benchmark

echo ""
echo "=== BENCHMARK RUST V1 NVIDIA RTX 3090 FP32 ==="
CUDA_VISIBLE_DEVICES=1 ./rust_component/target/release/examples/test_benchmark

echo ""
echo "=== BENCHMARK RUST V1 CPU AMD RYZEN FP32 ==="
FORCE_CPU=1 CUDA_VISIBLE_DEVICES="" ./rust_component/target/release/examples/test_benchmark

echo ""
echo "=== BENCHMARK RUST V2 NVIDIA RTX 4090 FP32 ==="
CUDA_VISIBLE_DEVICES=0 ./rust_component/target/release/examples/test_benchmark_v2

echo ""
echo "=== BENCHMARK RUST V2 NVIDIA RTX 3090 FP32 ==="
CUDA_VISIBLE_DEVICES=1 ./rust_component/target/release/examples/test_benchmark_v2

echo ""
echo "=== BENCHMARK RUST V2 CPU AMD RYZEN FP32 ==="
FORCE_CPU=1 CUDA_VISIBLE_DEVICES="" ./rust_component/target/release/examples/test_benchmark_v2

