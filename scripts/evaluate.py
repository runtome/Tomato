import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from src.configs.config import Config
from src.utils.seed import set_seed
from src.data.kfold import collect_all_samples, create_kfold_splits
from src.data.dataset import TomatoDataset
from src.data.transforms import get_val_transforms
from src.pipelines.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--fold", type=int, default=None, help="Fold to evaluate (default: all)")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.random_seed)

    val_transform = get_val_transforms(config.image_size)

    # Test set
    test_dir = os.path.join(config.dataset_path, "test")
    test_samples = collect_all_samples(test_dir)
    test_dataset = TomatoDataset(test_samples, transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # K-Fold splits for train/val evaluation
    train_dir = os.path.join(config.dataset_path, "train")
    all_samples = collect_all_samples(train_dir)
    splits = create_kfold_splits(all_samples, config.num_folds, config.random_seed, config.dataset_path)

    folds_to_run = [args.fold] if args.fold is not None else list(range(config.num_folds))

    fold_loaders = {}
    for fold in folds_to_run:
        train_idx, val_idx = splits[fold]
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        fold_loaders[fold] = {
            "train": DataLoader(
                TomatoDataset(train_samples, transform=val_transform),
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True,
            ),
            "val": DataLoader(
                TomatoDataset(val_samples, transform=val_transform),
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True,
            ),
        }

    evaluator = Evaluator(config)

    if len(folds_to_run) == 1:
        fold = folds_to_run[0]
        evaluator.evaluate_fold(
            fold, test_loader,
            train_loader=fold_loaders[fold]["train"],
            val_loader=fold_loaders[fold]["val"],
        )
    else:
        evaluator.evaluate_all_folds(folds_to_run, test_loader, fold_loaders)


if __name__ == "__main__":
    main()
