import argparse
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.configs.config import Config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.models.factory import create_model
from src.utils.checkpoint import load_checkpoint
from src.visualization.gradcam import generate_gradcam, save_gradcam_grid
from src.constants.labels import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use")
    parser.add_argument("--num_images", type=int, default=3, help="Images per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(args.seed)
    device = get_device()

    # Load model
    model = create_model(config.model_name, config.num_classes, pretrained=False)
    ckpt_path = os.path.join(
        "outputs", "models", config.save_name,
        f"{config.save_name}_fold_{args.fold}.pth",
    )
    load_checkpoint(model, ckpt_path, device=device)
    model = model.to(device)
    model.eval()

    test_dir = os.path.join(config.dataset_path, "test")
    save_dir = os.path.join("outputs", "models", config.save_name, "Grad-CAM")

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = sorted([
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ])
        selected = random.sample(files, min(args.num_images, len(files)))

        for i, fname in enumerate(selected):
            img_path = os.path.join(class_dir, fname)
            original, overlay = generate_gradcam(
                model, config.model_name, img_path, config.image_size, device,
            )
            save_path = os.path.join(save_dir, f"{class_name}_sample{i}.jpg")
            save_gradcam_grid(original, overlay, save_path)
            print(f"Saved: {save_path}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
