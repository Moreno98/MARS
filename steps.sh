#!/bin/bash
# MARS pipeline, shown for Gemma3-4B (layer 26).
# For another model, change --model, --target_layer_idx and --batch_size using the hyperparameters set in configs/<model>.sh.
# The batch sizes are configured for NVIDIA A100 64GB.
# Every command can be run as is from the repository root.

# Paths (see README, section Paths)
export MARS_WEIGHTS=/path/to/weights     # model weights, Hugging Face cache layout
export MARS_DATA=/path/to/datasets       # local datasets

# We flag as 'OPTIONAL' the steps that are not required as their output is already provided within this repo.
# Nonetheless, we provide the code for these steps for completeness.

#################### OPTIONAL: refusal directions and neutral mean ####################
# The text activations and the neutral image mean at the selected layers are provided in results/MARS.

# Text activations of prompts the model accepts (Alpaca)
python extract_directions.py --model Gemma3-4B --dataset Alpaca

# Text activations of prompts the model refuses (MaliciousInstruct)
python extract_directions.py --model Gemma3-4B --dataset MaliciousInstruct

# Mean activation of the 128 neutral images, used for centering
python collect_image_act.py --model Gemma3-4B --n_images 128

# NOTE: We also provide the script to generate the neutral images: generate_blank_dataset.py

#################### OPTIONAL: layer selection ####################
# The selected layers for each model are already provided within configs/<model>.sh.

# Text activations of the validation sets 
python extract_directions.py --model Gemma3-4B --dataset AlpacaLast --remove_check
python extract_directions.py --model Gemma3-4B --dataset HarmBenchContextual --remove_check
python extract_directions.py --model Gemma3-4B --dataset HarmBenchCopyright --remove_check

# Score layers and print the selected layer
python layer_selection/select_layers.py --models Gemma3-4B

# Plot the layer selection scores of all models (results/layer_selection/)
python layer_selection/plot_layer_selection.py

#################### Safety evaluation (MARS) ####################
# For the zero-shot model, replace --target_layer_idx 26 with --zero_shot.
# Reminder: Update batch size and target layer idx with the model specifications in their configs.

#### ViSU

# ViSU safe images
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Safe --batch_size 128 --deterministic --target_layer_idx 26

# E.g., for zero-shot:
# python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Safe --batch_size 128 --deterministic --zero_shot

# ViSU unsafe images
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26

#### HADES

# HADES 
python run_benchmark.py --model Gemma3-4B --inference_dataset HADES --batch_size 96 --deterministic --target_layer_idx 26 --neutral_prompt

# Safety Scores
python safety_score.py --evaluate_model Gemma3-4B --target_layer_idx 26 --direction_batch_size 96 --neutral_prompt

#### SIUO
python run_benchmark.py --model Gemma3-4B --inference_dataset SIUO --batch_size 96 --deterministic --target_layer_idx 26

#### HoliSafe
for subset in safe_safes_unsafe safe_unsafes unsafes unsafe_safes; do python run_benchmark.py --model Gemma3-4B --inference_dataset HoliSafe_$subset --batch_size 96 --deterministic --target_layer_idx 26; done

#### VideoSafetyBench, benign queries (video-capable models only). This step needs FFmpeg installed. 
#### Note: For Molmo2 use its specific environment.
for category in 1_Violent_Crimes 2_Non-Violent_Crimes 3_Sex-Related_Crimes 4_Child_Sexual_Exploitation 5_Defamation 6_Specificial_Advice 7_Privacy 8_Intellectual_Property 9_Indiscriminate_Weapons 10_Hate 11_Suicide_Self-Harm 12_Sexual_Content 13_Elections; do python run_inference.py --model Molmo2-8B --inference_dataset VideoSafetyBenchBenign_$category --inference_modality video --batch_size 1 --deterministic --target_layer_idx 22; done

#### VLGuard (US = safe_unsafes, UU = unsafes, and safe_safes)
for subset in safe_unsafes unsafes safe_safes; do python run_benchmark.py --model Gemma3-4B --inference_dataset VLGuard_$subset --batch_size 96 --deterministic --target_layer_idx 26; done

#### MM-SafetyBench, 13 categories x 4 splits
for category in EconomicHarm Financial_Advice Fraud Gov_Decision HateSpeech Health_Consultation Illegal_Activitiy Legal_Opinion Malware_Generation Physical_Harm Political_Lobbying Privacy_Violence Sex; do for split in SD SD_TYPO TYPO Text_only; do python run_benchmark.py --model Gemma3-4B --inference_dataset MMSafety_${category}_${split} --batch_size 1 --deterministic --target_layer_idx 26; done; done

#################### Utility evaluation (MARS) ####################

# MMMU, 30 subjects
for subject in Accounting Agriculture Architecture_and_Engineering Art Art_Theory Basic_Medical_Science Biology Chemistry Clinical_Medicine Computer_Science Design Diagnostics_and_Laboratory_Medicine Economics Electronics Energy_and_Power Finance Geography History Literature Manage Marketing Materials Math Mechanical_Engineering Music Pharmacy Physics Psychology Public_Health Sociology; do python run_benchmark.py --model Gemma3-4B --inference_dataset MMMUBench$subject --batch_size 1 --deterministic --target_layer_idx 26; done

# MMMU-Pro (4 options, 10 options, vision)
for subset in 4cls 10cls Vision; do python run_benchmark.py --model Gemma3-4B --inference_dataset MMMUBenchPro$subset --batch_size 1 --deterministic --target_layer_idx 26; done

# MMStar
python run_benchmark.py --model Gemma3-4B --inference_dataset MMStar --batch_size 96 --deterministic --target_layer_idx 26

#################### Ablations (ViSU and MMMU-Pro) ####################
# Each ablation changes only the steering arguments; shown here on ViSU unsafe, run it on the other datasets in the same way.

# Steering at all layers (no layer selection): drop --target_layer_idx
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic

# Without neutral-image centering
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26 --prober_mode ablation_no_centering

# Without the ReLU gate
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26 --prober_mode ablation_no_relu

# Neutral mean from fewer images (8, 24, 48 or 64): first collect the mean, then steer with it
python collect_image_act.py --model Gemma3-4B --n_images 8
python run_inference.py --model Gemma3-4B --inference_dataset ViSU_Vision_Unsafe --batch_size 128 --deterministic --target_layer_idx 26 --n_images_mean 8

#################### Tables ####################

# Print all paper tables from results/ ("–" for missing experiments)
python tables/make_tables.py
