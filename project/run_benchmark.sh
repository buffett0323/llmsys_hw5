#!/bin/bash
# Problem 1.3: Run benchmarks and generate plots
# Run from repo root: bash project/run_benchmark.sh

cd "$(dirname "$0")/.."
N_EPOCHS=5

echo "=== Running Single GPU benchmark (world_size=1, batch_size=64) ==="
python project/run_data_parallel.py --world_size 1 --batch_size 64 --benchmark_only --skip_first_epoch --n_epochs $N_EPOCHS --benchmark_output benchmark_single_gpu.json

echo ""
echo "=== Running 2 GPUs benchmark (world_size=2, batch_size=128) ==="
python project/run_data_parallel.py --world_size 2 --batch_size 128 --benchmark_only --skip_first_epoch --n_epochs $N_EPOCHS --benchmark_output benchmark_dual_gpu.json

echo ""
echo "=== Generating plots ==="
python project/plot.py ./workdir ./submit_figures

echo ""
echo "Done! Figures saved to submit_figures/"
