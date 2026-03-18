#!/bin/bash
# Problem 2.3: Run pipeline vs model parallel benchmarks and generate plots
# Run from repo root: bash project/run_pipeline_benchmark.sh

cd "$(dirname "$0")/.."
N_EPOCHS=3

echo "=== Running Model Parallel benchmark ==="
python project/run_pipeline.py --model_parallel_mode='model_parallel' --benchmark_only --skip_first_epoch --n_epochs $N_EPOCHS --benchmark_output benchmark_model_parallel.json

echo ""
echo "=== Running Pipeline Parallel benchmark ==="
python project/run_pipeline.py --model_parallel_mode='pipeline_parallel' --benchmark_only --skip_first_epoch --n_epochs $N_EPOCHS --benchmark_output benchmark_pipeline_parallel.json

echo ""
echo "=== Generating pipeline plots ==="
python project/plot.py ./workdir ./submit_figures pp

echo ""
echo "Done! Figures saved to submit_figures/pp_training_time.png and pp_tokens_per_second.png"
