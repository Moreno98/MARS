# Text-only layer selection (Sec. 4.3). The selected layers are already set in configs/<model>.sh.
import os, sys, csv, argparse
import torch
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils.utils import get_results_dir

MODELS = {
    'Gemma3-4B': 'gemma3',
    'Molmo2-8B': 'molmo2',
    'InternVL3_5_8B': 'internvl3.5',
    'Qwen3-VL-8B': 'qwen3-vl',
    'Qwen3-5-9B': 'qwen3.5',
}
DATASETS = ['Alpaca', 'MaliciousInstruct', 'AlpacaLast', 'HarmBenchContextual', 'HarmBenchCopyright']
METRICS = ['direction_consistency', 'cont_separation', 'cop_separation', 'safe_text_margin']

def normalize(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-07)

def nd_accuracy(safe, unsafe):
    mu_s, mu_u = safe.mean(0), unsafe.mean(0)
    frac_safe = ((safe - mu_s).norm(dim=-1) < (safe - mu_u).norm(dim=-1)).float().mean().item()
    frac_unsafe = ((unsafe - mu_u).norm(dim=-1) < (unsafe - mu_s).norm(dim=-1)).float().mean().item()
    return (frac_safe + frac_unsafe) / 2

def score_layer(acc, rej, safe_val, unsafe_cont, unsafe_cop):
    train_direction = normalize(rej.mean(dim=0) - acc.mean(dim=0))
    val_direction = normalize(unsafe_cont.mean(dim=0) - safe_val.mean(dim=0))
    boundary = ((normalize(acc) @ train_direction).mean() + (normalize(rej) @ train_direction).mean()) / 2
    return {
        'direction_consistency': (train_direction @ val_direction).item(),
        'cont_separation': nd_accuracy(safe_val, unsafe_cont),
        'cop_separation': nd_accuracy(safe_val, unsafe_cop),
        'safe_text_margin': (boundary - (normalize(safe_val) @ train_direction).mean()).item(),
    }

def activations_dir(root, dataset, model):
    for sub in ['directions_percentage/text/post_instruction/directions_1.0/heads_False', 'heads_False']:
        path = os.path.join(root, dataset, model, sub)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f'No activations for {dataset}/{model} in {root}')

def load(root, dataset, model):
    path = activations_dir(root, dataset, model)
    n_layers = len([d for d in os.listdir(path) if d.isdigit()])
    return [torch.load(os.path.join(path, str(l), 'output_direction_full.pt'), map_location='cpu', weights_only=False).float() for l in range(n_layers)]

def config_layer(key):
    with open(os.path.join(REPO, 'configs', f'{key}.sh')) as f:
        for line in f:
            if line.startswith('LAYER='):
                return int(line.split('=')[1].split()[0])

def select(root, model, min_layer):
    acc, rej, safe_val, cont, cop = [load(root, d, model) for d in DATASETS]
    metrics = {}
    for l in range(min_layer, len(acc)):
        mean = torch.cat([acc[l], rej[l]], dim=0).mean(dim=0)
        metrics[l] = score_layer(acc[l] - mean, rej[l] - mean, safe_val[l] - mean, cont[l] - mean, cop[l] - mean)
    # min-max normalise each metric across layers
    minmax = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-8)
    for k in METRICS:
        v = minmax(np.array([metrics[l][k] for l in metrics]))
        for l, x in zip(metrics, v):
            metrics[l][k] = x
    # context separability: the two HarmBench subsets combined into a single score
    separation = minmax(np.array([metrics[l]['cont_separation'] + metrics[l]['cop_separation'] for l in metrics]))
    for l, s in zip(metrics, separation):
        metrics[l]['score'] = metrics[l]['direction_consistency'] + s + metrics[l]['safe_text_margin']
    return max(metrics, key=lambda l: metrics[l]['score']), metrics, len(acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    parser.add_argument('--activations_root', default=os.path.join(get_results_dir(), 'MARS'))
    parser.add_argument('--out_dir', default=os.path.join(get_results_dir(), 'layer_selection'))
    parser.add_argument('--min_layer', type=int, default=20)
    opt = parser.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)

    print('| Model | Layers | Selected | Config |\n|---|---|---|---|')
    for model in opt.models:
        best, metrics, n_layers = select(opt.activations_root, model, opt.min_layer)
        with open(os.path.join(opt.out_dir, f'{model}_scores.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer'] + METRICS + ['score'])
            for l in metrics:
                writer.writerow([l] + [metrics[l][k] for k in METRICS + ['score']])
        print(f'| {model} | {n_layers} | {best} | {config_layer(MODELS[model])} |')

if __name__ == '__main__':
    main()
