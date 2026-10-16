import os
from utils import argparse
from torch.utils.data import DataLoader
import utils.probers as probers
import json

def main():
    opt = argparse.run_benchmark_parser()
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

    report, acc = dataset.forward_fn(model, loader)

    with open(os.path.join(opt['results_dir'], "report.jsonl"), "w") as f:
        for item in report:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(opt['results_dir'], "accuracy.txt"), "w") as f:
        if isinstance(acc, tuple):
            f.write(f'Refusal on Safe Samples: {acc[0]}\n')
            f.write(f'Refusal on Unsafe Samples: {acc[1]}\n')
        else:
            f.write(f'Accuracy: {acc}')
   
    print('Evaluation Completed.')
    if isinstance(acc, tuple):
        print(f'Dataset: {opt["inference_dataset"]} Model: {opt["model"]}\nRefusal on Safe Samples: {acc[0]}%\nRefusal on Unsafe Samples: {acc[1]}%')
    else:
        print(f"Dataset: {opt['inference_dataset']} Model: {opt['model']} Accuracy: {acc}%")
    print(f"Report saved: {os.path.join(opt['results_dir'], 'report.jsonl')}")
    print(f"Accuracy saved: {os.path.join(opt['results_dir'], 'accuracy.txt')}")

if __name__ == "__main__":
    main()
