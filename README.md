<div align="center">

# Harnessing Textual Refusal Directions for Multimodal Safety
[[`Abstract`](https://arxiv.org/abs/2606.31876)][[`pdf`](https://arxiv.org/pdf/2606.31876)][[`arXiv`](https://arxiv.org/abs/2606.31876)][[`BibTeX`](#bibtex)]

[Moreno D`Incà](https://scholar.google.com/citations?user=tdTJsOMAAAAJ&hl=en), [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en), [Massimiliano Mancini](https://mancinimassimiliano.github.io/)

</div>

<div style="text-align: justify">

>**Abstract:** Safety in Multimodal Large Language Models (MLLMs) depends on unsafe multimodal data for post-training alignment or activation steering. However, such data is hard to collect, must be gathered anew for each modality, and its collection and distribution raise ethical and legal concerns. In this work, we relax this constraint and investigate whether textual refusal directions, extracted directly from the LLM backbone, generalize across modalities (i.e., image, video). Preliminary findings confirm this ability, though effectiveness is conditioned by layer selection, steering strength, and cross-modal alignment, with the latter causing safe multimodal inputs to be spuriously steered toward refusal. Building on this, we introduce Modality-Agnostic Refusal Steering (MARS), a lightweight training-free approach that injects multimodal safety without the need for multimodal safety data. MARS corrects modality misalignment via activation centering, adaptively scales steering strength within a geometrically defined trust region, and selects the optimal intervention layer, operating at the first generated token. Evaluated on five SOTA MLLMs across safety, utility, and video jailbreak, MARS achieves consistent safety gains while preserving utility up to over-refusal. MARS reveals that safety-relevant structure is shared across modalities and that textual refusal directions are a powerful and underexplored foundation for multimodal alignment.
</div>


## Installation

>**Disclaimer**: The datasets used within this repository contain unsafe and potentially disturbing content. This repository is intended for research purposes only.

We recommend to use a virtual environment. We provide two requirements files, with one specific for Molmo2 (which requires an older `transformers`).

```bash
# Create a virtual environment and activate it
uv venv
source .venv/bin/activate
# Install the required packages 
uv pip install -r requirements.txt
```

Install from `requirements_molmo2.txt` in a separate environment for Molmo2.

Videos are decoded with `torchcodec`, which needs the FFmpeg shared libraries (versions 4 to 7), please make sure to install them.

Flash attention is required within both environments. This code has been tested with flash attention version `2.8.3+cu126torch2.10`. You may install it with `pip install flash-attn==2.8.3 --no-build-isolation` or from pre-build wheels (see [prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)).

Before running any code, the required model weights and datasets must be downloaded and placed in the correct folders. See the [Paths](#paths) below for details.

### Paths

All paths are set with two environment variables. No file in the repository needs to be edited.

| Variable | Content | Default |
|---|---|---|
| `MARS_WEIGHTS` | Model weights (Hugging Face cache layout) | `./weights` |
| `MARS_DATA` | Local datasets | `./datasets` |

```bash
export MARS_WEIGHTS=/path/to/weights
export MARS_DATA=/path/to/datasets
```

#### Model weights

The code loads each model from `$MARS_WEIGHTS/models--<org>--<name>/snapshots/<revision>` (see `model_configs` in `models/models.py` and `LLAMA_GUARD_PATH` in `utils/utils.py`). 

Downloading with `--cache-dir` creates exactly this layout:

```bash
hf download google/gemma-3-4b-it                --revision 093f9f388b31de276ce2de164bdc2081324b9767 --cache-dir $MARS_WEIGHTS
hf download allenai/Molmo2-8B                   --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b --cache-dir $MARS_WEIGHTS
hf download OpenGVLab/InternVL3_5-8B-HF         --revision 741a7d03020411e666c6109218ab71e08151ef86 --cache-dir $MARS_WEIGHTS
hf download Qwen/Qwen3-VL-8B-Instruct           --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --cache-dir $MARS_WEIGHTS
hf download Qwen/Qwen3.5-9B                     --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --cache-dir $MARS_WEIGHTS
hf download meta-llama/Llama-Guard-3-8B         --revision 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 --cache-dir $MARS_WEIGHTS   # For safety score
```

If you already have the weights elsewhere, either point `MARS_WEIGHTS` to that cache or change `model_name` in `models/models.py` to your folder.

#### Datasets

Provide the required datasets under `$MARS_DATA`. You may download some of them by using:

```bash
hf download etri-vilab/holisafe-bench --repo-type dataset --revision b6892a0efadd275026bc903953158a9943d76028 --cache-dir $MARS_DATA
hf download Lin-Chen/MMStar           --repo-type dataset --revision bc98d668301da7b14f648724866e57302778ab27 --cache-dir $MARS_DATA
hf download sinwang/SIUO              --repo-type dataset --revision 024e80a01795376b9fed12f8073a12f2275f22ee --cache-dir $MARS_DATA
hf download ys-zong/VLGuard           --repo-type dataset --revision b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46 --cache-dir $MARS_DATA
```

The VLGuard test images (`test.zip`) are extracted automatically to `$MARS_DATA/VLGuard/test` on the first run.

Datasets to place manually under `$MARS_DATA`:

```bash
$MARS_DATA/
  ViSU/ViSU-Text_test.json                   # ViSU text annotations (test set)
  ViSU/unsafe_images/<id>/0.jpg              # ViSU unsafe images
  COCO/images/{train2017,val2017}/           # COCO 2017 images (safe ViSU images)
  videosafetybench/video/<category>/...      # VideoSafetyBench videos
```

Note: unsafe ViSU images are not publicly available. Refer to the [ViSU official page](https://huggingface.co/datasets/aimagelab/ViSU-Text) or to the ViSU paper to generate them.

MMMU, MMMU-Pro, MM-SafetyBench, HADES, Alpaca, MaliciousInstruct, HarmBench and the VideoSafetyBench are downloaded automatically from the Hugging Face Hub on the first run (and placed in the hugging face cache).

Some of the above datasets require hf authentication (`hf auth login`).

## Usage

This repo provides the code to run zero-shot inference and MARS.

We flag as `[OPTIONAL]` the steps that are **not required** as the repo comes with their outputs already precomputed. Nonetheless, we provide the full code for completeness.

The code supports the following steps:
- `[OPTIONAL]` Refusal directions extraction and neutral image mean collection (once per model)
- `[OPTIONAL]` Layer Selection (once per model)
- Safety Evaluation on ViSU, HADES, SIUO, HoliSafe, VLGuard, MM-SafetyBench, and VideoSafetyBench 
- Utility Evaluation on MMMU, MMMU-Pro, and MMStar
- Ablation studies and further analysis

Hyperparameters for each model are under `configs/<model>.sh`. Please, check them for the correct layer and batch size for each model. Batch size is based on a NVIDIA A100 64GB GPU. The results of each inference will be placed under the `results` folder. We provide examples to run each step on Gemma3. The same commands can be run for other models by changing the `--model` argument alongside the hyperparameters. For Molmo2, make sure to use its specific environment.

For the zero-shot performance, replace `--target_layer_idx` with `--zero_shot`.

Model aliases and intervention layer (i.e., output of the layer selection step):

| Model | `--model` | Intervention Layer | Config |
|---|---|---|---|
| Gemma3-4B | `Gemma3-4B` | 26 | `configs/gemma3.sh` |
| Molmo2-8B | `Molmo2-8B` | 22 | `configs/molmo2.sh` |
| InternVL3.5-8B | `InternVL3_5_8B` | 25 | `configs/internvl3.5.sh` |
| Qwen3-VL-8B | `Qwen3-VL-8B` | 23 | `configs/qwen3-vl.sh` |
| Qwen3.5-9B | `Qwen3-5-9B` | 22 | `configs/qwen3.5.sh` |

Refer to [`steps.sh`](steps.sh) for a complete example of all steps for Gemma3.

### `[OPTIONAL]` Refusal directions and neutral image mean

The text activations and the neutral image mean at the selected layers are provided in results/MARS.

For completeness, we provide the commands to run this step.

```bash
# Text activations of prompts the model accepts (Alpaca)
python extract_directions.py --model Gemma3-4B --dataset Alpaca

# Text activations of prompts the model refuses (MaliciousInstruct)
python extract_directions.py --model Gemma3-4B --dataset MaliciousInstruct

# Mean activation of the 128 neutral images, used for centering
python collect_image_act.py --model Gemma3-4B --n_images 128
```

NOTE: Neutral images are already provided under `neutral_images/`. Nevertheless, we provide the script to generate random neutral images: `generate_blank_dataset.py`.

### `[OPTIONAL]` Layer Selection

The selected layers are already provided in `configs/`, so this step is not required. 

Layer selection uses text activations only and requires the following datasets: AlpacaLast (safe), and HarmBench contextual and copyright (unsafe).

Example usage for Gemma3-4B:

```bash
# Text activations of the validation sets 
python extract_directions.py --model Gemma3-4B --dataset AlpacaLast --remove_check
python extract_directions.py --model Gemma3-4B --dataset HarmBenchContextual --remove_check
python extract_directions.py --model Gemma3-4B --dataset HarmBenchCopyright --remove_check

# Score layers and print the selected layer
python layer_selection/select_layers.py --models Gemma3-4B

# Plot the layer selection scores of all models (results/layer_selection/)
python layer_selection/plot_layer_selection.py
```

### Safety evaluation

```bash
# ViSU safe and unsafe images
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Safe --batch_size 128 --deterministic --target_layer_idx 26
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26

# HADES and safety score
python run_benchmark.py --model Gemma3-4B --inference_dataset HADES --batch_size 96 --deterministic --target_layer_idx 26 --neutral_prompt
python safety_score.py --evaluate_model Gemma3-4B --target_layer_idx 26 --direction_batch_size 96 --neutral_prompt

# SIUO (HoliSafe_<subset> and VLGuard_<subset> run the same way, see steps.sh)
python run_benchmark.py --model Gemma3-4B --inference_dataset SIUO --batch_size 96 --deterministic --target_layer_idx 26
```

MM-SafetyBench (`MMSafety_<category>_<split>`) and VideoSafetyBench (`VideoSafetyBenchBenign_<category>`, with `--inference_modality video`) loop over their categories, see [`steps.sh`](steps.sh).

Gemma3 does not support video input, so VideoSafetyBench is only evaluated on Molmo2, InternVL3.5, Qwen3-VL, and Qwen3.5.

### Utility evaluation

```bash
# MMStar
python run_benchmark.py --model Gemma3-4B --inference_dataset MMStar --batch_size 96 --deterministic --target_layer_idx 26
```

MMMU (`MMMUBench<subject>`) and MMMU-Pro (`MMMUBenchPro4cls`, `MMMUBenchPro10cls`, `MMMUBenchProVision`) use `--batch_size 1`, see [`steps.sh`](steps.sh).

### Ablations

Ablations can be run with the following commands (ViSU unsafe with Gemma3-4B).

```bash
# All layers (no layer selection)
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic

# Without centering
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26 --prober_mode ablation_no_centering
# Without ReLU
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26 --prober_mode ablation_no_relu

# Neutral mean from fewer images (8, 24, 48, 64)
python collect_image_act.py --model Gemma3-4B --n_images 8
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26 --n_images_mean 8
```

### Tables

```bash
python tables/make_tables.py   # create tables from results/ ("–" for missing experiments)
```

## BibTeX

If you find this work useful for your research, please cite it:

```bibtex
@misc{dinca2026harnessingtextualrefusaldirections,
      title={Harnessing Textual Refusal Directions for Multimodal Safety}, 
      author={Moreno D'Incà and Nicu Sebe and Massimiliano Mancini},
      year={2026},
      eprint={2606.31876},
      archivePrefix={arXiv},
}
```