import torch, os
import utils.config as config
import random

LLAMA_GUARD_PATH = os.path.join('models--meta-llama--Llama-Guard-3-8B', 'snapshots', '7327bd9f6efbbe6101dc6cc4736302b3cbb6e425')

def ensure_bhwc(x):
    # x: [B, ?, ?, ?]
    x = x.squeeze()
    if x.ndim == 4:
        raise ValueError(f"Expected 3D tensor, got shape {x.shape}")

    if x.shape[0] in (1, 3):
        return x.permute(1, 2, 0)  # CHW -> HWC

    # If channels already last
    elif x.shape[-1] in (1, 3):
        return x  # already HWC

    else:
        raise ValueError(f"Cannot infer channel dimension from shape {x.shape}")

def set_deterministic(seed):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_weights_dir():
    # Directory containing the HF-cache style model folders (models--<org>--<name>/snapshots/<hash>)
    return os.environ.get('MARS_WEIGHTS', os.path.join(os.getcwd(), 'weights'))

def get_data_dir():
    # Directory containing the local datasets (ViSU, COCO, generated images, blank images, HF snapshots)
    return os.environ.get('MARS_DATA', os.path.join(os.getcwd(), 'datasets'))

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_results_dir():
    return os.environ.get('MARS_RESULTS', os.path.join(REPO_DIR, 'results'))

def count_refused_responses(responses):
    if isinstance(responses, list):
        count = 0
        for response in responses:
            if is_refused_response(response):
                count += 1
        return count
    elif isinstance(responses, str):
        return 1 if is_refused_response(responses) else 0
    else:
        raise ValueError("Responses must be a string or a list of strings.")

def is_refused_response(response):
    if isinstance(response, str):
        for template in config.REFUSAL_TEMPLATES:
            if template.lower() in response.lower():
                return True
        return False
    else:
        raise ValueError("Response must be a string.")
