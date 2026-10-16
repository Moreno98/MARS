import os, re, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import utils.config as config
import utils.benchmark_utils as bench_utils

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS = os.environ.get('MARS_RESULTS', os.path.join(ROOT, 'results'))
MODEL_KEYS = ['gemma3', 'molmo2', 'internvl3.5', 'qwen3-vl', 'qwen3.5']
TAIL = os.path.join('safe_perc_1.0_unsafe_perc_1.0', 'heads_False')

def load_config(key):
    cfg = {}
    for line in open(os.path.join(ROOT, 'configs', f'{key}.sh')):
        m = re.match(r'^(\w+)="?([^"\s#]+)"?', line)
        if m:
            cfg[m.group(1)] = m.group(2)
    return cfg

CFG = {k: load_config(k) for k in MODEL_KEYS}

def run_dir(dataset, cfg, bs, method, modality='image', suffix='', n_images=128):
    # dataset: '<results_dir>/<dataset>' for benchmarks, '<dataset>' for ViSU and video
    if method == 'zero_shot':
        return os.path.join(RESULTS, 'original', dataset, modality, cfg['MODEL'], f'deterministic_batch-size-{bs}{suffix}')
    mode, layer = {
        'mars': ('mars', cfg['LAYER']),
        'all_layers': ('mars', 'all'),
        'no_centering': ('ablation_no_centering', cfg['LAYER']),
        'no_relu': ('ablation_no_relu', cfg['LAYER']),
    }[method]
    return os.path.join(RESULTS, 'MARS', dataset, cfg['MODEL'], 'refusal_removal_results', modality, 'text',
                        'Alpaca_MaliciousInstruct', 'post_instruction', f'BlankImages_safe_{n_images}', mode,
                        f'layer_id_{layer}', f'deterministic_batch-size-{bs}_alpha-1.0{suffix}', TAIL)

def last_number(path, pattern):
    if not os.path.exists(path):
        return None
    values = re.findall(pattern, open(path).read())
    return float(values[-1]) if values else None

def refusal_txt(d):
    return last_number(os.path.join(d, 'refusal_removal_results.txt'), r'\(([\d.]+)%\)')

def accuracy_txt(d):
    return last_number(os.path.join(d, 'accuracy.txt'), r'Accuracy: ([\d.]+)')

def safety_txt(d):
    return last_number(os.path.join(d, 'safety_score_llama-guard-v4.txt'), r'Safety score: ([\d.]+)')

def read_report(d):
    path = os.path.join(d, 'report.jsonl')
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path) if l.strip()]

def report_accuracy(d):
    rows = read_report(d)
    if not rows:
        return None
    by_id = {r.get('id', i): str(r['correct_pred']).lower() == 'true' for i, r in enumerate(rows)}
    return sum(by_id.values()) / len(by_id) * 100

def mmstar_score(d):
    rows = read_report(d)
    return None if not rows else bench_utils.compute_mmstar_scores(rows)['final score'] * 100

def mean(values):
    return None if not values or any(v is None for v in values) else sum(values) / len(values)

def fmt(v):
    return '–' if v is None else f'{v:.1f}'

def print_table(title, columns, rows):
    print(f'\n### {title}\n')
    print('| Method | ' + ' | '.join(columns) + ' |')
    print('|' + '---|' * (len(columns) + 1))
    for name, values in rows:
        print(f'| {name} | ' + ' | '.join(fmt(v) for v in values) + ' |')

def per_model(fn, methods=('zero_shot', 'mars')):
    return [(m, [x for k in MODEL_KEYS for x in fn(CFG[k], m)]) for m in methods]

def names(sub):
    return [f"{CFG[k]['MODEL']} {s}" for k in MODEL_KEYS for s in sub]

def visu(cfg, m, n_images=128):
    return [refusal_txt(run_dir(d, cfg, cfg['BS_VISU'], m, n_images=n_images)) for d in ('ViSU_Vision_Safe', 'ViSU_Vision_Unsafe')]

def hades(cfg, m):
    d = run_dir('HADES/HADES', cfg, cfg['BS_SAFETY'], m, suffix='_neutral-prompt')
    return [accuracy_txt(d), safety_txt(d)]

def siuo(cfg, m):
    return [accuracy_txt(run_dir('SIUO/SIUO', cfg, cfg['BS_SAFETY'], m))]

def holisafe(cfg, m):
    subsets = [accuracy_txt(run_dir(f'HoliSafe/HoliSafe_{s}', cfg, cfg['BS_SAFETY'], m)) for s in ('safe_safes_unsafe', 'safe_unsafes', 'unsafes', 'unsafe_safes')]
    return [subsets[0], mean(subsets)]

def vlguard(cfg, m):
    us = accuracy_txt(run_dir('VLGuard/VLGuard_safe_unsafes', cfg, cfg['BS_SAFETY'], m))
    uu = accuracy_txt(run_dir('VLGuard/VLGuard_unsafes', cfg, cfg['BS_SAFETY'], m))
    return [us, uu, mean([us, uu])]

def mmsafetybench(cfg, m):
    return [mean([accuracy_txt(run_dir(f'MMSafetyBench/MMSafety_{c}_{s}', cfg, 1, m)) for c in config.MMSafetyBenchCategories for s in config.MMSafetyBenchSplits])]

def mmmu_pro(cfg, m, n_images=128):
    return [mean([report_accuracy(run_dir(f'MMMUPro/{d}', cfg, 1, m, n_images=n_images)) for d in ('MMMUBenchPro4cls', 'MMMUBenchPro10cls', 'MMMUBenchProVision')])]

def utility(cfg, m):
    mmmu = mean([report_accuracy(run_dir(f'MMMU/MMMUBench{c}', cfg, 1, m)) for c in config.MMMU_CATEGORIES])
    return [mmmu] + mmmu_pro(cfg, m)

def mmstar(cfg, m):
    return [mmstar_score(run_dir('MMStar/MMStar', cfg, cfg['BS_MMSTAR'], m))]

def video(cfg, m):
    if cfg['MODEL'] == 'Gemma3-4B':
        return [None]
    return [mean([refusal_txt(run_dir(f'VideoSafetyBenchBenign_{c}', cfg, 1, m, modality='video')) for c in sorted(config.VideoSafetyBenchCategories, key=lambda x: int(x.split('_')[0]))])]

if __name__ == '__main__':
    print(f'Results root: {RESULTS}')
    print_table('ViSU: refusal (%) on safe (lower) and unsafe (higher) images', names(['Safe', 'Unsafe']), per_model(visu))
    print_table('HADES (neutral prompt): refusal and Llama Guard 3 safety (%)', names(['Refusal', 'Safety']), per_model(hades))
    print_table('SIUO: refusal (%)', names(['']), per_model(siuo))
    print_table('HoliSafe: refusal (%) on SSU and on all unsafe settings', names(['SSU', 'Overall']), per_model(holisafe))
    print_table('Utility: accuracy (%) on MMMU and MMMU-Pro (mean over categories)', names(['MMMU', 'Pro']), per_model(utility))
    print_table('VideoSafetyBench (benign queries): refusal (%), mean over 13 categories', names(['']), per_model(video))
    print_table('MMSafetyBench: refusal (%), mean over 52 subsets', names(['']), per_model(mmsafetybench))
    print_table('VLGuard: refusal (%)', names(['US', 'UU', 'Mean']), per_model(vlguard))
    print_table('MMStar: accuracy (%)', names(['']), per_model(mmstar))
    print_table('Ablation: ViSU refusal and MMMU-Pro accuracy', names(['Safe', 'Unsafe', 'Pro']),
                per_model(lambda c, m: visu(c, m) + mmmu_pro(c, m), methods=('zero_shot', 'mars', 'all_layers', 'no_centering', 'no_relu')))
    print_table('Ablation: number of neutral images', names(['Safe', 'Unsafe', 'Pro']),
                [(f'{n} images', [x for k in MODEL_KEYS for x in visu(CFG[k], 'mars', n) + mmmu_pro(CFG[k], 'mars', n)]) for n in (8, 24, 48, 64, 128)])
