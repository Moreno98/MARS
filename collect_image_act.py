from utils import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils.probers as probers

def main():
    opt = argparse.collect_img_act()
    model = opt['model_config']['class'](
        model_name=opt['model_config']['model_id'],
        cache_activations=True,
        tokens_type=opt['tokens_type'],
        perc_token_id=opt['perc_token_id'],
        modality='image',
        do_sample=True # Only one token is generated, activations are collected on the prompt
    )
    dataset = opt['dataset_config']['class'](**opt['dataset_config'])

    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=8)
    prober = getattr(probers, opt['prober'])(
        model=model.module_to_probe,
        mode='collect',
        base_dir=opt['activations_dir'],
        heads=opt['heads'],
        collect_type="image"
    )

    for idx, data in enumerate(tqdm(loader)):
        safe_image = data.unsqueeze(0)
        assert len(safe_image.shape) == 4, "Batch size does not match"
        _ = model(safe_image, max_new_tokens=1)

    print(f"Act saved: {opt['activations_dir']}")

    prober.save_img_activations(opt['image_type'], opt['n_images'])


if __name__ == "__main__":
    main()
