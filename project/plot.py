import json
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
from pathlib import Path

def plot_bar(means, stds, labels, fig_name, ylabel='Value'):
    """Plot bar chart with error bars."""
    fig, ax = plt.subplots()
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds,
           align='center', alpha=0.7, ecolor='black', capsize=10, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150)
    plt.close(fig)

def plot_data_parallel_benchmark(workdir='./workdir', outdir='./submit_figures'):
    """
    Load benchmark results and create 2 figures:
    1. Training time comparison (Single GPU vs Data Parallel per device)
    2. Tokens per second comparison (Single GPU vs Data Parallel throughput)
    """
    workdir = Path(workdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Look for benchmark files (run with --benchmark_output to control names)
    single_file = workdir / 'benchmark_single_gpu.json'
    dual_file = workdir / 'benchmark_dual_gpu.json'
    # Also check auto-generated names
    if not single_file.exists():
        single_file = workdir / 'benchmark_w1_bs64.json'
    if not dual_file.exists():
        dual_file = workdir / 'benchmark_w2_bs128.json'

    single_data = None
    dual_data = None
    if single_file.exists():
        single_data = json.load(open(single_file))
    if dual_file.exists():
        dual_data = json.load(open(dual_file))

    # Fallback: scan for any benchmark files
    if single_data is None or dual_data is None:
        benchmark_files = list(workdir.glob('benchmark_*.json'))
        for bf in benchmark_files:
            if 'aggregate' in bf.name:
                continue
            data = json.load(open(bf))
            ws = data.get('world_size', 0)
            if ws == 1 and single_data is None:
                single_data = data
            elif ws == 2 and dual_data is None:
                dual_data = data

    if single_data is None and dual_data is None:
        print("No benchmark data found. Run benchmarks first:")
        print("  python project/run_data_parallel.py --world_size 1 --batch_size 64 --benchmark_only --skip_first_epoch --n_epochs 5 --benchmark_output benchmark_single_gpu.json")
        print("  python project/run_data_parallel.py --world_size 2 --batch_size 128 --benchmark_only --skip_first_epoch --n_epochs 5 --benchmark_output benchmark_dual_gpu.json")
        return

    # Figure 1: Training Time (seconds)
    means_t, stds_t, labels_t = [], [], []
    if dual_data and 'per_rank' in dual_data:
        for r in dual_data['per_rank']:
            means_t.append(r['mean_time'])
            stds_t.append(r['std_time'])
            labels_t.append(f'Data Parallel - GPU{r["rank"]}')
    elif dual_data:
        means_t.append(dual_data['training_time_mean'])
        stds_t.append(dual_data['training_time_std'])
        labels_t.append('Data Parallel (2 GPUs)')
    if single_data:
        means_t.append(single_data['training_time_mean'])
        stds_t.append(single_data['training_time_std'])
        labels_t.append('Single GPU')

    if means_t:
        plot_bar(means_t, stds_t, labels_t,
                 outdir / 'training_time.png',
                 ylabel='Training Time (Second)')
        print(f"Saved {outdir}/training_time.png")

    # Figure 2: Tokens Per Second (throughput)
    # For single: one bar. For 2 GPUs: show combined throughput
    means_tok, stds_tok, labels_tok = [], [], []
    if dual_data and 'per_rank' in dual_data:
        # Show per-device tokens/sec and combined
        tok_sum = sum(r['mean_tokens_per_sec'] for r in dual_data['per_rank'])
        tok_std = np.sqrt(sum(r['std_tokens_per_sec']**2 for r in dual_data['per_rank']))
        means_tok.append(tok_sum)
        stds_tok.append(tok_std)
        labels_tok.append('Data Parallel (2 GPUs)')
    elif dual_data:
        means_tok.append(dual_data['tokens_per_sec_mean'])
        stds_tok.append(dual_data['tokens_per_sec_std'])
        labels_tok.append('Data Parallel (2 GPUs)')
    if single_data:
        means_tok.append(single_data['tokens_per_sec_mean'])
        stds_tok.append(single_data['tokens_per_sec_std'])
        labels_tok.append('Single GPU')

    if means_tok:
        plot_bar(means_tok, stds_tok, labels_tok,
                 outdir / 'tokens_per_second.png',
                 ylabel='Tokens Per Second')
        print(f"Saved {outdir}/tokens_per_second.png")

# Fill the data points here
if __name__ == '__main__':
    import sys
    workdir = sys.argv[1] if len(sys.argv) > 1 else './workdir'
    outdir = sys.argv[2] if len(sys.argv) > 2 else './submit_figures'
    # Resolve paths relative to project dir
    project_dir = Path(__file__).resolve().parent
    parent_dir = project_dir.parent
    workdir = parent_dir / workdir.lstrip('./')
    outdir = parent_dir / outdir.lstrip('./')
    plot_data_parallel_benchmark(workdir=workdir, outdir=outdir)
