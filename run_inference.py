import os
from utils import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils.probers as probers
import utils.utils as utils

def main():
    opt = argparse.run_inference_parser()
    model = opt['model_config']['class'](
        model_name=opt['model_config']['model_id'],
        modality=opt['inference_modality'],
        do_sample=not opt['deterministic']
    )
    dataset = opt['inference_dataset_config']['class'](**opt['inference_dataset_config'])
    loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    if opt['prober'] is not None:
        prober = getattr(probers, opt['prober'])(
            model=model.module_to_probe,
            mode='apply',
            base_dir=None,
            direction_type=opt['prober_mode'],
            safe_direction_dir=opt['safe_directions_dir'],
            unsafe_direction_dir=opt['unsafe_directions_dir'],
            target_layer_idx=opt['target_layer_idx'],
            act_mean_path = opt['act_mean_path'],
            n_images_mean = opt['n_images_mean'],
            act_mean_type = opt['act_mean_type'],
            target_token_id = opt['target_token'],
            coeff_alpha = opt['coeff_alpha']
        )

    refused_answers = 0
    dataset_size = len(loader.dataset)
    responses = []
    for idx, data in enumerate(tqdm(loader)):
        if opt['inference_modality'] == 'image':
            image_path = data[0]
            image = data[1]

            outputs = model(image, max_new_tokens=256)

            if opt['batch_size'] is not None:
                for img_path, model_response in zip(image_path, outputs):
                    responses.append((img_path, model_response))
                    refused_answers += utils.count_refused_responses(model_response)
        elif opt['inference_modality'] == 'video':
            ids, questions, video_paths, categories, subcategories = data
            outputs = model(video_paths, text=questions, max_new_tokens=256)

            if opt['batch_size'] is not None:
                for id, question, video_path, category, subcategory, model_response in zip(ids, questions, video_paths, categories, subcategories, outputs):
                    responses.append(
                        (
                            id, 
                            question,
                            video_path, 
                            category, 
                            subcategory,
                            model_response
                        )
                    )
                    refused_answers += utils.count_refused_responses(model_response)

    refused_accuracy = (refused_answers/dataset_size)*100
    print(f"Refused answers: {refused_answers}/{dataset_size} ({refused_accuracy:.2f}%)")

    with open(os.path.join(opt['results_dir'], 'refusal_removal_results.txt'), 'w') as f:
        f.write(f"Refused answers: {refused_answers}/{dataset_size} ({refused_accuracy:.2f}%)\n")

    print(opt['results_dir'])
    with open(os.path.join(opt['results_dir'], 'responses.txt'), 'w') as f:
        for data in responses:
            if isinstance(data, tuple) or isinstance(data, list):
                if len(data) == 2:
                    image_path, response = data
                    f.write(f"{image_path} | {response}\n\n\n")
                else:
                    id, question, video_path, category, subcategory, model_response = data
                    f.write(f"{id} | {question} | {video_path} | {category} | {subcategory} | {model_response}\n\n\n")
    
    with open(os.path.join(opt['results_dir'], 'responses_cleaned.txt'), 'w') as f:
        for data in responses:
            if isinstance(data, tuple) or isinstance(data, list):
                if len(data) == 2:
                    image_path, response = data
                    response = response.replace('\n', ' ')
                    f.write(f"{image_path} | {response}\n")
                else:
                    id, question, video_path, category, subcategory, model_response = data
                    model_response = model_response.replace('\n', ' ')
                    f.write(f"{id} | {question} | {video_path} | {category} | {subcategory} | {model_response}\n\n\n")

if __name__ == "__main__":
    main()
