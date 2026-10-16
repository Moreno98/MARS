import os
import random
import argparse
from PIL import Image

def generate_blank_images(
    output_dir="dataset",
    num_images=100,
    width=256,
    height=256,
    seed=0
):
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        print(f"'{output_dir}' already contains images, not overwriting.")
        return
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    for i in range(num_images):
        # Random RGB color
        color = tuple(rng.randint(0, 255) for _ in range(3))

        # Create image
        img = Image.new("RGB", (width, height), color)

        # Save image
        filename = f"img_{i:05d}.png"
        img.save(os.path.join(output_dir, filename))

    print(f"Generated {num_images} images in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate randomly colored (neutral) images.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neutral_images'))
    parser.add_argument("--num_images", type=int, default=128)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_blank_images(
        output_dir=args.output_dir,
        num_images=args.num_images,
        width=args.size,
        height=args.size,
        seed=args.seed
    )
