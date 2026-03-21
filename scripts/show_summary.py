import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torchinfo import summary
from src.configs.config import Config
from src.models.factory import create_model
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Show model summary")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = get_device()

    model = create_model(config.model_name, config.num_classes, pretrained=False)
    model = model.to(device)

    print(f"\nModel: {config.model_name}")
    print(f"Image size: {config.image_size}")
    print(f"Device: {device}\n")

    summary(model, input_size=(1, 3, config.image_size, config.image_size), device=device)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
