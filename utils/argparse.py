import argparse
import utils.utils as utils
from . import datasets
import os
from models import models

utils.set_deterministic(0)

PROBERS = ["MARS"]
PROBER_MODES = ["mars", "ablation_no_relu", "ablation_no_centering"]

def extract_directions_parser():
    parser = argparse.ArgumentParser(description="Extract textual refusal directions from a model.")
    parser.add_argument("--model", type=str, required=True, help="Model to extract refusal directions from.", choices=list(models.model_configs.keys()))
    parser.add_argument("--prober", type=str, default="MARS", help="Prober to use for extraction.", choices=PROBERS)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for extraction.", choices=["Alpaca", "MaliciousInstruct", "AlpacaLast", "HarmBenchContextual", "HarmBenchCopyright"])
    parser.add_argument("--perc_token_id", type=float, default=1.0, help="Percentage of token ID to use for extraction. E.g., 1 is the full subpart.")
    parser.add_argument("--tokens_type", type=str, default="post_instruction", help="Type of tokens to consider.", choices=["post_instruction"])
    parser.add_argument("--modality", type=str, default="text", choices=["text"], help="Modality to use.")
    parser.add_argument("--remove_check", action='store_true', help="Keep all prompts, without checking whether they are accepted / refused (layer selection validation sets).")
    opt = vars(parser.parse_args())
    opt['heads'] = False
    opt['model_config'] = models.model_configs[opt['model']]
    opt['dataset_config'] = datasets.DATASET_REGISTRY[opt['dataset']]

    opt['model_config']['model_id'] = os.path.join(utils.get_weights_dir(), opt['model_config']['model_name'])
    opt['results_dir'] = os.path.join(utils.get_results_dir(), opt['prober'], opt['dataset'], opt['model'])
    opt['directions_dir'] = os.path.join(opt['results_dir'], 'directions_percentage', opt['modality'], opt['tokens_type'], f'directions_{opt["perc_token_id"]}', f'heads_{opt["heads"]}')
    os.makedirs(opt['directions_dir'], exist_ok=True)
    return opt

def collect_img_act():
    parser = argparse.ArgumentParser(description="Collect the neutral image activation mean.")
    parser.add_argument("--model", type=str, required=True, help="Model to collect activations from.", choices=list(models.model_configs.keys()))
    parser.add_argument("--prober", type=str, default="MARS", help="Prober to use for extraction.", choices=PROBERS)
    parser.add_argument("--dataset", type=str, default="BlankImages", help="Dataset to use for extraction.", choices=["BlankImages"])
    parser.add_argument("--perc_token_id", type=float, default=1.0, help="Percentage of token ID to use for extraction. E.g., 1 is the full subpart.")
    parser.add_argument("--tokens_type", type=str, default="post_instruction", help="Type of tokens to consider.", choices=["post_instruction"])
    parser.add_argument("--image_type", type=str, default="safe", choices=['safe'], help="Type of images to collect.")
    parser.add_argument("--n_images", type=int, default=128, help="Number of images to collect.")
    opt = vars(parser.parse_args())
    opt['heads'] = False
    opt['model_config'] = models.model_configs[opt['model']]
    opt['dataset_config'] = dict(datasets.DATASET_REGISTRY[opt['dataset']])
    opt['dataset_config']['subset'] = opt['n_images']

    opt['model_config']['model_id'] = os.path.join(utils.get_weights_dir(), opt['model_config']['model_name'])
    opt['results_dir'] = os.path.join(utils.get_results_dir(), opt['prober'], opt['dataset'], opt['model'])
    opt['activations_dir'] = os.path.join(opt['results_dir'], 'image_activations', opt['tokens_type'], f'token_perc_{opt["perc_token_id"]}', f'heads_{opt["heads"]}')
    os.makedirs(opt['activations_dir'], exist_ok=True)
    return opt

def add_inference_args(parser, modalities):
    parser.add_argument("--model", type=str, required=True, help="Model to use.", choices=list(models.model_configs.keys()))
    parser.add_argument("--zero_shot", action='store_true', help="Run the original model without MARS.")
    parser.add_argument("--prober", type=str, help="Prober to use.", default="MARS", choices=PROBERS)
    parser.add_argument("--prober_mode", type=str, help="Steering method: mars, or the ablations ablation_no_relu, ablation_no_centering.", default="mars", choices=PROBER_MODES)
    parser.add_argument("--target_layer_idx", nargs='*', type=int, help="Layer where to apply the direction. If not given, all layers are steered.", default=None)
    parser.add_argument("--safe_dataset", type=str, default="Alpaca", help="Apply safe directions extracted from this dataset.", choices=["Alpaca", "AlpacaOriginal"])
    parser.add_argument("--unsafe_dataset", type=str, default="MaliciousInstruct", help="Apply unsafe directions extracted from this dataset.", choices=["MaliciousInstruct", "MaliciousInstructOriginal"])
    parser.add_argument("--inference_dataset", type=str, required=True, help="Apply MLLM on this dataset.", choices=list(datasets.DATASET_REGISTRY.keys()))
    parser.add_argument("--safe_perc_token_id", type=float, default=1.0, help="Percentage of token ID to use for application. E.g., 1 is full sentence.")
    parser.add_argument("--unsafe_perc_token_id", type=float, default=1.0, help="Percentage of token ID to use for application. E.g., 1 is full sentence.")
    parser.add_argument("--tokens_type", type=str, default="post_instruction", help="Type of tokens to consider.", choices=["post_instruction"])
    parser.add_argument("--coeff_alpha", type=float, default=1.0, help="Alpha.")
    parser.add_argument("--batch_size", type=int, default=None, help="Inference Batch Size.")
    parser.add_argument("--inference_modality", type=str, default="image", choices=modalities, help="Modality to test the directions on.")
    parser.add_argument("--direction_modality", type=str, default="text", choices=["text"], help="Direction modality to use.")
    parser.add_argument("--deterministic", action='store_true', default=False, help="Deterministic.")
    parser.add_argument("--exp_name", type=str, default=None, help="Specific name for the experiment.")
    parser.add_argument("--n_images_mean", default=128, type=int, help="Number of neutral images used for the mean.")
    parser.add_argument("--dataset_mean", default="BlankImages", type=str, help="Dataset mean")
    parser.add_argument("--act_mean_type", default="safe", type=str, choices=['safe'], help="Activation mean type")
    parser.add_argument("--target_token", default=-1, type=int, help="Target token id where to apply the method. Default is -1.")
    parser.add_argument("--neutral_prompt", action='store_true', help="Use the neutral HADES activity prompt instead of the dataset instruction.")
    parser.add_argument("--num_shards", default=None, type=int, help="Number of dataset shards.")
    parser.add_argument("--shard_idx", default=None, type=int, help="Dataset shard index.")

def finalize_inference_opt(opt):
    opt['heads'] = False
    if opt['zero_shot']:
        opt['prober'] = None

    if (opt['num_shards'] is None) != (opt['shard_idx'] is None):
        raise ValueError("Both num_shards and shard_idx must be specified for sharded inference.")
    if opt['num_shards'] is not None:
        if opt['num_shards'] < 1:
            raise ValueError("num_shards must be >= 1.")
        if opt['shard_idx'] < 0 or opt['shard_idx'] >= opt['num_shards']:
            raise ValueError("shard_idx must be in [0, num_shards - 1].")

    opt['model_config'] = models.model_configs[opt['model']]
    opt['inference_dataset_config'] = dict(datasets.DATASET_REGISTRY[opt['inference_dataset']])
    opt['inference_dataset_config']['neutral_prompt'] = opt['neutral_prompt']
    opt['inference_dataset_config']['num_shards'] = opt['num_shards']
    opt['inference_dataset_config']['shard_idx'] = opt['shard_idx']

    opt['model_config']['model_id'] = os.path.join(utils.get_weights_dir(), opt['model_config']['model_name'])
    results_base = utils.get_results_dir()

    inference_dataset_results_path = opt['inference_dataset']
    if opt['inference_dataset_config'].get('results_dir', None) is not None:
        inference_dataset_results_path = os.path.join(opt['inference_dataset_config']['results_dir'], opt['inference_dataset'])

    alpha_txt = f"_alpha-{opt['coeff_alpha']}" if (opt['coeff_alpha'] and opt['prober'] is not None) else ''
    exp_name_default = f"deterministic_batch-size-{opt['batch_size']}{alpha_txt}" if opt['deterministic'] else f"non-deterministic_batch-size-{opt['batch_size']}{alpha_txt}"
    exp_name = opt['exp_name'] if opt['exp_name'] is not None else exp_name_default
    if opt['neutral_prompt']:
        exp_name = f"{exp_name}_neutral-prompt"
    if opt['num_shards'] is not None:
        exp_name = f"{exp_name}_shard-{opt['shard_idx']}-of-{opt['num_shards']}"

    if opt['prober'] is None:
        opt['results_dir'] = os.path.join(
            results_base,
            'original',
            inference_dataset_results_path,
            opt['inference_modality'],
            opt['model'],
            exp_name
        )
    else:
        opt['directions_dir_base'] = os.path.join(results_base, opt['prober'])

        image_mean_str = f"{opt['dataset_mean']}_{opt['act_mean_type']}_{opt['n_images_mean']}"
        opt['act_mean_path'] = os.path.join(
            opt['directions_dir_base'],
            opt['dataset_mean'],
            opt['model'],
            "image_activations",
            opt["tokens_type"],
            f"token_perc_{opt['safe_perc_token_id']}",
            "heads_False"
        )

        opt['safe_directions_dir'] = os.path.join(opt['directions_dir_base'], opt['safe_dataset'], opt['model'], 'directions_percentage', opt['direction_modality'], opt['tokens_type'], f'directions_{opt["safe_perc_token_id"]}', f'heads_{opt["heads"]}')
        opt['unsafe_directions_dir'] = os.path.join(opt['directions_dir_base'], opt['unsafe_dataset'], opt['model'], 'directions_percentage', opt['direction_modality'], opt['tokens_type'], f'directions_{opt["unsafe_perc_token_id"]}', f'heads_{opt["heads"]}')

        if opt['target_layer_idx'] is None:
            layer_id = 'all'
        else:
            layer_id = '-'.join([str(l) for l in opt['target_layer_idx']])

        opt['results_dir'] = os.path.join(
            opt['directions_dir_base'],
            inference_dataset_results_path,
            opt['model'],
            'refusal_removal_results',
            opt['inference_modality'],
            opt['direction_modality'],
            f"{opt['safe_dataset']}_{opt['unsafe_dataset']}",
            opt['tokens_type'],
            image_mean_str,
            opt['prober_mode'],
            f'layer_id_{layer_id}',
            exp_name,
            f'safe_perc_{opt["safe_perc_token_id"]}_unsafe_perc_{opt["unsafe_perc_token_id"]}',
            f'heads_{opt["heads"]}'
        )

    os.makedirs(opt['results_dir'], exist_ok=True)
    return opt

def run_inference_parser():
    parser = argparse.ArgumentParser(description="Run MARS (or zero-shot) on ViSU and VideoSafetyBench.")
    add_inference_args(parser, ["image", "video"])
    return finalize_inference_opt(vars(parser.parse_args()))

def run_benchmark_parser():
    parser = argparse.ArgumentParser(description="Run MARS (or zero-shot) on a benchmark.")
    add_inference_args(parser, ["image"])
    return finalize_inference_opt(vars(parser.parse_args()))

def run_safety_score():
    parser = argparse.ArgumentParser(description="Compute the Llama Guard 3 safety score on HADES.")
    parser.add_argument("--evaluate_model", type=str, required=True, help="Model to evaluate.")
    parser.add_argument("--zero_shot", action='store_true', help="Score the original model outputs.")
    parser.add_argument("--prober", type=str, help="Prober used.", default="MARS", choices=PROBERS)
    parser.add_argument("--prober_mode", type=str, help="Steering method.", default="mars", choices=PROBER_MODES)
    parser.add_argument("--safe_dataset", type=str, default="Alpaca", choices=["Alpaca", "AlpacaOriginal"])
    parser.add_argument("--unsafe_dataset", type=str, default="MaliciousInstruct", choices=["MaliciousInstruct", "MaliciousInstructOriginal"])
    parser.add_argument("--inference_dataset", type=str, default="HADES", choices=["HADES"])
    parser.add_argument("--safe_perc_token_id", type=float, default=1.0)
    parser.add_argument("--unsafe_perc_token_id", type=float, default=1.0)
    parser.add_argument("--tokens_type", type=str, default="post_instruction", choices=["post_instruction"])
    parser.add_argument("--inference_modality", type=str, default="image", choices=["image"])
    parser.add_argument("--direction_modality", type=str, default="text", choices=["text"])
    parser.add_argument("--n_images_mean", default=128, type=int)
    parser.add_argument("--dataset_mean", default="BlankImages", type=str)
    parser.add_argument("--act_mean_type", default="safe", type=str, choices=['safe'])
    parser.add_argument("--exp_name", type=str, default=None, help="Specific name for the experiment.")
    parser.add_argument("--target_layer_idx", nargs='*', type=int, help="Layer where the direction was applied.", default=None)
    parser.add_argument("--coeff_alpha", type=float, default=1.0, help="Alpha.")
    parser.add_argument("--direction_batch_size", type=int, default=None, help="Batch Size used in the original experiment.")
    parser.add_argument("--neutral_prompt", action='store_true', help="Use the neutral HADES activity prompt setting.")
    parser.add_argument("--llama_guard_prompt_variant", type=str, default="v4", choices=["v4"], help="Prompt variant used when scoring HADES with Llama Guard.")

    opt = vars(parser.parse_args())
    opt['heads'] = False
    if opt['zero_shot']:
        opt['prober'] = None
    opt['inference_dataset_config'] = dict(datasets.DATASET_REGISTRY[opt['inference_dataset']])
    opt['inference_dataset_config']['neutral_prompt'] = opt['neutral_prompt']
    opt['model_id'] = os.path.join(utils.get_weights_dir(), utils.LLAMA_GUARD_PATH)
    results_base = utils.get_results_dir()

    alpha_txt = f"_alpha-{opt['coeff_alpha']}" if (opt['coeff_alpha'] and opt['prober'] is not None) else ''
    exp_name_default = f"deterministic_batch-size-{opt['direction_batch_size']}{alpha_txt}"
    exp_name = opt['exp_name'] if opt['exp_name'] is not None else exp_name_default
    if opt['neutral_prompt']:
        exp_name = f"{exp_name}_neutral-prompt"

    inference_dataset_results_path = opt['inference_dataset']
    if opt['inference_dataset_config'].get('results_dir', None) is not None:
        inference_dataset_results_path = os.path.join(opt['inference_dataset_config']['results_dir'], opt['inference_dataset'])

    if opt['prober'] is None:
        opt['results_dir'] = os.path.join(
            results_base,
            'original',
            inference_dataset_results_path,
            opt['inference_modality'],
            opt['evaluate_model'],
            exp_name
        )
    else:
        if opt['target_layer_idx'] is None:
            layer_id = 'all'
        else:
            layer_id = '-'.join([str(l) for l in opt['target_layer_idx']])

        image_mean_str = f"{opt['dataset_mean']}_{opt['act_mean_type']}_{opt['n_images_mean']}"
        opt['results_dir'] = os.path.join(
            results_base,
            opt['prober'],
            inference_dataset_results_path,
            opt['evaluate_model'],
            'refusal_removal_results',
            opt['inference_modality'],
            opt['direction_modality'],
            f"{opt['safe_dataset']}_{opt['unsafe_dataset']}",
            opt['tokens_type'],
            image_mean_str,
            opt['prober_mode'],
            f'layer_id_{layer_id}',
            exp_name,
            f'safe_perc_{opt["safe_perc_token_id"]}_unsafe_perc_{opt["unsafe_perc_token_id"]}',
            f'heads_{opt["heads"]}'
        )

    opt['safety_score_filename'] = f"safety_score_llama-guard-{opt['llama_guard_prompt_variant']}.txt"
    opt['guard_answers_filename'] = f"guard_answers_llama-guard-{opt['llama_guard_prompt_variant']}.jsonl"
    return opt
