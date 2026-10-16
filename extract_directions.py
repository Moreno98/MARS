from utils import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils.probers as probers
import os
import utils.utils as utils

def check_refusal(refusal, dataset, prober):
    correct_behavior = (dataset.safe and refusal == 0) or (not dataset.safe and refusal == 1)
    if correct_behavior:
        prober.accept_forward()
    else:
        prober.clear_tmp()

def main():
    opt = argparse.extract_directions_parser()
    model = opt['model_config']['class'](
        model_name=opt['model_config']['model_id'],
        cache_activations=True,
        tokens_type=opt['tokens_type'],
        perc_token_id=opt['perc_token_id'],
        modality=opt['modality'],
        do_sample=False
    )
    dataset = opt['dataset_config']['class'](**opt['dataset_config'])
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2)
    prober = getattr(probers, opt['prober'])(
        model=model.module_to_probe,
        mode='collect',
        base_dir=opt['directions_dir'],
        heads=opt['heads'],
        collect_type='text',
        save_full=True
    )

    max_new_tokens = 256
    if opt['remove_check']:
        max_new_tokens = 1

    refused_prompts = 0
    responses = []
    for idx, data in enumerate(tqdm(loader)):
        if type(data) == list and len(data) == 1:
            data = data[0]
        response = model(data, max_new_tokens=max_new_tokens)

        if isinstance(response, list):
            response = response[0]
        
        refusal = utils.count_refused_responses(response)
        if not opt['remove_check']:
            check_refusal(refusal, dataset, prober)
        else:
            prober.accept_forward()
        
        refused_prompts += refusal
        responses.append(
            (data, response, refusal)
        )

    prober.save_activations()
    refused_accuracy = (refused_prompts/len(dataset))*100
    print(f"Refused answers: {refused_prompts}/{len(dataset)} ({refused_accuracy:.2f}%)")

    with open(os.path.join(opt['results_dir'], 'refusal_results.txt'), 'w') as f:
        f.write(f"Refused answers: {refused_prompts}/{len(dataset)} ({refused_accuracy:.2f}%)\n")

    print(opt['results_dir'])
    with open(os.path.join(opt['results_dir'], 'responses.txt'), 'w') as f:
        for prompt, response, refusal in responses:
            f.write(f"{prompt} | {response}\n\n Refusal is {bool(refusal)}\n--------\n\n")
    
    with open(os.path.join(opt['results_dir'], 'responses_cleaned.txt'), 'w') as f:
        for prompt, response, _ in responses:
            f.write(f"{prompt} | {response.replace('\n', ' ')}\n")


if __name__ == "__main__":
    main()
