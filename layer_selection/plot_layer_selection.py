# Layer selection scores across models (reads the CSVs written by select_layers.py)
import os, sys, csv, shutil, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, MaxNLocator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils.utils import get_results_dir

NAMES = {
    'Gemma3-4B': 'Gemma3',
    'Molmo2-8B': 'Molmo2',
    'InternVL3_5_8B': 'InternVL 3.5',
    'Qwen3-VL-8B': 'Qwen3-VL',
    'Qwen3-5-9B': 'Qwen3.5',
}

USETEX = shutil.which('latex') is not None
plt.rcParams.update({'text.usetex': USETEX, 'font.family': 'serif'})

def tt(s):
    return rf'\texttt{{{s}}}' if USETEX else s

ROWS = [
    ('direction_consistency', tt('Dir.') + '\n' + tt('consistency')),
    ('separation', tt('Cont.') + '\n' + tt('Separability')),
    ('safe_text_margin', tt('Safe txt margin')),
    ('score', tt('Final score')),
]

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    layers = np.array([int(r['layer']) for r in rows])
    v = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'layer'}
    sep = v['cont_separation'] + v['cop_separation']
    v['separation'] = (sep - sep.min()) / (sep.max() - sep.min() + 1e-8)
    v['score'] = v['direction_consistency'] + v['separation'] + v['safe_text_margin']
    return layers, v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_dir', default=os.path.join(get_results_dir(), 'layer_selection'))
    opt = parser.parse_args()

    models = [m for m in NAMES if os.path.exists(os.path.join(opt.scores_dir, f'{m}_scores.csv'))]
    fig, axes = plt.subplots(len(ROWS), len(models), figsize=(12.8, 8.4), sharex='col', squeeze=False)
    for col, model in enumerate(models):
        layers, values = load(os.path.join(opt.scores_dir, f'{model}_scores.csv'))
        start = max(16, layers[0] + (-layers[0]) % 4)
        for row, (key, label) in enumerate(ROWS):
            ax = axes[row][col]
            ax.plot(layers, values[key], marker='o', markersize=3, linewidth=1.4, color=plt.get_cmap('tab10')(row))
            ax.axhline(0, linestyle='--', linewidth=0.8, color='0.25', alpha=0.8)
            if key == 'score':
                best = int(np.argmax(values[key]))
                ax.scatter(layers[best], values[key][best], marker='*', s=140, color='gold', edgecolors='black', linewidths=0.8, zorder=10)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + 0.14)
            ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
            ax.xaxis.set_major_locator(FixedLocator(np.arange(start, layers[-1] + 1, 4)))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.35)
            ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.25)
            ax.tick_params(which='major', direction='in', length=4, width=0.8, top=True, right=True, labelsize=11)
            ax.tick_params(which='minor', direction='in', length=2, width=0.6, top=True, right=True)
            if col == 0:
                ax.set_ylabel(label, fontsize=12, labelpad=8)
            if row == 0:
                ax.set_title(rf'\textsc{{{NAMES[model]}}}' if USETEX else NAMES[model], fontsize=16, pad=8)
            if row == len(ROWS) - 1:
                ax.set_xlabel(tt('Layers'), fontsize=16, labelpad=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.09, top=0.94, hspace=0.30, wspace=0.16)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(opt.scores_dir, f'layer_selection.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.03)
    print(f'Saved {opt.scores_dir}/layer_selection.pdf')

if __name__ == '__main__':
    main()
