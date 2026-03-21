import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.prepare_dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset with CLIP-based deduplication")
    parser.add_argument("--source-train", type=str, default="tomato/train", help="Source train directory")
    parser.add_argument("--source-valid", type=str, default="tomato/valid", help="Source valid directory")
    parser.add_argument("--dest", type=str, default="datasets", help="Destination directory")
    parser.add_argument("--train-per-class", type=int, default=1000, help="Train images per class")
    parser.add_argument("--test-per-class", type=int, default=200, help="Test images per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clip-threshold", type=float, default=0.95, help="CLIP similarity threshold for merging")
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP-based merging (filename grouping only)")
    args = parser.parse_args()

    prepare_dataset(
        source_train_dir=args.source_train,
        source_valid_dir=args.source_valid,
        dest_dir=args.dest,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        clip_threshold=args.clip_threshold,
        use_clip=not args.no_clip,
    )


if __name__ == "__main__":
    main()
