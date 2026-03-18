import sys
from pathlib import Path

cousin_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(cousin_dir))

from functools import partial
import time
import os
import argparse
import tqdm
import json
import datasets
import numpy as np
from transformers import AutoConfig, GPT2LMHeadModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import torch.distributed as dist
from torch.multiprocessing import Process

from data_parallel.dataset import partition_dataset
from utils import get_tokenizer, evaluate_bleu, save_grad_weights, collate_batch, evaluate_loss, generate, train

PYTEST = False

def average_gradients(model):
    '''Aggregate the gradients from different GPUs
    
    1. Iterate through the parameters of the model 
    2. Use `torch.distributed` package and call the reduce fucntion to aggregate the gradients of all the parameters
    3. Average the gradients over the world_size (total number of devices)
    '''
    # BEGIN_HW5_1_2
    # raise NotImplementedError("Data Parallel Not Implemented Yet")
    world_size = dist.get_world_size()
    for name, param in model.named_parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
    # END_HW5_1_2

def setup(rank, world_size, backend):
    '''Setup Process Group

    1. Set the environment variables `MASTER_ADDR` as `localhost` or `127.0.0.1`  and `MASTER_PORT` as `11868`
    2. Use `torch.distributed` to init the process group
    '''
    # BEGIN_HW5_1_2
    # raise NotImplementedError("Data Parallel Not Implemented Yet")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11868'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # END_HW5_1_2


def run_dp(
    rank, world_size, backend,
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=128,
    n_epochs=10,
    batch_size=128,
    learning_rate=1e-4,
    benchmark_only=False,
    max_batches=0,
    pytest_mode=False,
    skip_first_epoch=False):
    workdir = f'./workdir'
    os.makedirs(workdir, exist_ok=True)

    config = AutoConfig.from_pretrained('gpt2')
    config.save_pretrained(workdir)
    
    ### Distributed Training Setup
    setup(rank, world_size, backend)
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(rank)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = {
        split: datasets.load_dataset(dataset_name, trust_remote_code=True, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    ### MAKE SMALLER
    dataset['train'] = dataset['train'][:5000]
    dataset['validation'] = dataset['validation'][:1000]
    dataset['test'] = dataset['test'][:100]
    ###

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config.vocab_size,
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        device=rank)
    
    ### Get Partition of the Training Dataset on Device {rank}
    train_loader = partition_dataset(rank, world_size, dataset['train'], batch_size=batch_size, collate_fn=collate_fn)

    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_time = []
    total_tokens_per_sec = []

    for epoch_idx in range(n_epochs):
        desc = f'rank {rank}/{world_size} epoch {epoch_idx}/{n_epochs}'

        start = time.time()
        avg_tokens_per_sec, _  = train(
                                    model=model,
                                    optimizer=optimizer,
                                    examples=train_loader,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    desc=desc,
                                    rank=rank,
                                    average_gradients_fn=average_gradients,
                                    max_batches=max_batches)
        end = time.time()
        if not pytest_mode:
            training_time = end - start
            print(f'Epoch {epoch_idx} on Rank {rank}: Training Time = {training_time}, Tokens_per_sec = {avg_tokens_per_sec}')
            total_time.append(training_time)
            total_tokens_per_sec.append(avg_tokens_per_sec)

            if not benchmark_only:
                validation_loss = evaluate_loss(
                    model=model,
                    examples=val_loader,
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    desc=desc)

                print(f'Epoch {epoch_idx} on Rank {rank}: Validation Loss = {validation_loss}')

                gen_sents = generate(
                    model=model,
                    examples=dataset['test'],
                    src_key=src_key,
                    tgt_key=tgt_key,
                    tokenizer=tokenizer,
                    model_max_length=model_max_length,
                    device=rank,
                    desc=desc)

                gen_examples = []
                for example, gen_sent in zip(dataset['test'], gen_sents):
                    gen_examples.append({'example': example, 'gen': gen_sent})
                json.dump(gen_examples, open(
                    f'{workdir}/rank{rank}_gen_epoch{epoch_idx}.json', 'w'), indent=4)

                eval_scores = evaluate_bleu(
                    examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
                print(f'Epoch {epoch_idx} on Rank {rank}: {eval_scores}')

                json.dump(
                    {'validation_loss': validation_loss, **eval_scores, 'training_time': training_time, 'tokens_per_sec': avg_tokens_per_sec},
                    open(f'{workdir}/rank{rank}_results_epoch{epoch_idx}.json', 'w'))
        elif pytest_mode:
            save_grad_weights(model, rank)
            break
    dist.destroy_process_group()
    if not pytest_mode:
        # Drop first epoch for warmup if requested
        if skip_first_epoch and len(total_time) > 1:
            total_time = total_time[1:]
            total_tokens_per_sec = total_tokens_per_sec[1:]
        # You only get the average training time and tokens_per_second per device
        # To compute the throughput, you need to sum up the tokens_per_sec across all the devices based on epochs
        print(f'Rank {rank} training time: avg:{np.mean(total_time)}, std:{np.std(total_time)}, \
        tokens_per_second: avg: {np.mean(total_tokens_per_sec)}, std:{np.std(total_tokens_per_sec)}')
        # Save benchmark results for plotting (benchmark_only mode)
        if benchmark_only:
            benchmark_path = Path(workdir) / f'rank{rank}_benchmark.json'
            json.dump({
                'rank': rank,
                'world_size': world_size,
                'batch_size': batch_size,
                'training_times': total_time,
                'tokens_per_sec': total_tokens_per_sec,
                'mean_time': float(np.mean(total_time)),
                'std_time': float(np.std(total_time)) if len(total_time) > 1 else 0.0,
                'mean_tokens_per_sec': float(np.mean(total_tokens_per_sec)),
                'std_tokens_per_sec': float(np.std(total_tokens_per_sec)) if len(total_tokens_per_sec) > 1 else 0.0,
            }, open(benchmark_path, 'w'), indent=2)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pytest', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='bbaaaa/iwslt14-de-en-preprocess')
    parser.add_argument('--model_max_length', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--benchmark_only', action='store_true', help='Run only training benchmarks, skip validation/generation')
    parser.add_argument('--skip_first_epoch', action='store_true', help='Drop first epoch from metrics (warmup)')
    parser.add_argument('--benchmark_output', type=str, default='', help='Save benchmark aggregate to workdir/<name>.json')
    parser.add_argument('--max_batches', type=int, default=0, help='Max batches per epoch (0=full epoch)')
    args = parser.parse_args()
    if args.pytest:
        PYTEST = True
    else:
        PYTEST = False

    processes = []

    '''Create Process to start distributed training

    Hint:
    1. You can use Process from torch.multiprocessing to define the process
    2. You should start the processes to work and terminate resources properly
    '''
    # BEGIN_HW5_1_3
    world_size = args.world_size
    backend = 'gloo' if world_size == 1 else 'nccl'

    for rank in range(world_size):
        p = Process(target=run_dp, args=(rank, world_size, backend, args.dataset, args.model_max_length, args.n_epochs, args.batch_size, args.learning_rate, args.benchmark_only, args.max_batches, args.pytest, args.skip_first_epoch))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Aggregate benchmark results for plotting
    if args.benchmark_only and not args.pytest:
        workdir = Path('./workdir')
        benchmark_files = sorted(workdir.glob('rank*_benchmark.json'))
        if benchmark_files:
            results = [json.load(open(f)) for f in benchmark_files]
            world_size = results[0]['world_size']
            batch_size = results[0]['batch_size']
            if world_size == 1:
                agg = {
                    'training_time_mean': results[0]['mean_time'],
                    'training_time_std': results[0]['std_time'],
                    'tokens_per_sec_mean': results[0]['mean_tokens_per_sec'],
                    'tokens_per_sec_std': results[0]['std_tokens_per_sec'],
                    'config': 'single_gpu',
                    'world_size': world_size,
                    'batch_size': batch_size,
                }
            else:
                # Average training time across devices, sum tokens_per_sec for throughput
                time_means = [r['mean_time'] for r in results]
                time_stds = [r['std_time'] for r in results]
                tok_means = [r['mean_tokens_per_sec'] for r in results]
                tok_stds = [r['std_tokens_per_sec'] for r in results]
                agg = {
                    'training_time_mean': float(np.mean(time_means)),
                    'training_time_std': float(np.sqrt(np.mean([s**2 for s in time_stds]))) if time_stds else 0.0,
                    'tokens_per_sec_mean': float(np.sum(tok_means)),
                    'tokens_per_sec_std': float(np.sqrt(np.sum([s**2 for s in tok_stds]))) if tok_stds else 0.0,
                    'config': f'{world_size}_gpus',
                    'world_size': world_size,
                    'batch_size': batch_size,
                    'per_rank': results,
                }
            out_name = args.benchmark_output or f'benchmark_w{world_size}_bs{batch_size}.json'
            out_path = workdir / out_name
            json.dump(agg, open(out_path, 'w'), indent=2)
            print(f'Benchmark results saved to {out_path}')
    # END_HW5_1_3