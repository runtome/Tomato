import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.configs.config import Config
from src.visualization.augmentation_viz import visualize_augmentations


def main():
    parser = argparse.ArgumentParser(description="Visualize data augmentations")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num_augmented", type=int, default=5, help="Number of augmented versions")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    save_path = os.path.join("outputs", "models", config.save_name, "augmentation_viz.png")

    visualize_augmentations(
        dataset_path=config.dataset_path,
        image_size=config.image_size,
        num_augmented=args.num_augmented,
        seed=config.random_seed,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
