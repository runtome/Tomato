import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from src.configs.config import Config
from src.utils.seed import set_seed
from src.data.kfold import collect_all_samples, create_kfold_splits
from src.data.dataset import TomatoDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.pipelines.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a model with K-Fold CV")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--fold", type=int, default=None, help="Fold to train (default: all)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.random_seed)

    train_dir = os.path.join(config.dataset_path, "train")
    samples = collect_all_samples(train_dir)
    splits = create_kfold_splits(samples, config.num_folds, config.random_seed, config.dataset_path)

    folds_to_run = [args.fold] if args.fold is not None else list(range(config.num_folds))

    train_transform = get_train_transforms(config.image_size)
    val_transform = get_val_transforms(config.image_size)

    for fold in folds_to_run:
        set_seed(config.random_seed)

        train_idx, val_idx = splits[fold]
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_dataset = TomatoDataset(train_samples, transform=train_transform)
        val_dataset = TomatoDataset(val_samples, transform=val_transform)

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers, pin_memory=True,
        )

        trainer = Trainer(config)
        trainer.fit(train_loader, val_loader, fold)
        trainer.cleanup()


if __name__ == "__main__":
    main()
