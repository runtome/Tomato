"""
Dataset preparation with CLIP-based deduplication.

Strategy:
1. Filter only 256x256 images
2. Group images by filename prefix (strip augmentation suffixes like _flipLR, _180deg, etc.)
3. Compute CLIP embeddings for ungrouped images and merge near-duplicates (cosine sim > threshold)
4. Select 1000 images for train by picking groups — keeps augmented variants together
5. Select 200 images for test the same way
6. Save group mapping so k-fold can use StratifiedGroupKFold to prevent leakage
"""

import os
import re
import json
import shutil
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.constants.labels import CLASS_NAMES

# Regex to strip known augmentation suffixes from filenames
AUG_SUFFIX_PATTERN = re.compile(
    r"_(flipLR|flipTB|180deg|90deg|270deg|new30degFlipLR|"
    r"change_180|change_90|change_270|flip)(?=\.\w+$)",
    re.IGNORECASE,
)


def _strip_aug_suffix(filename):
    """Remove augmentation suffix to get the base image name."""
    name, ext = os.path.splitext(filename)
    cleaned = AUG_SUFFIX_PATTERN.sub("", name + ext)
    # Return just the name part (without ext) as group key
    return os.path.splitext(cleaned)[0]


def _filter_256(directory):
    """Return list of filenames that are 256x256."""
    valid = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath)
            if img.size == (256, 256):
                valid.append(fname)
        except Exception:
            continue
    return valid


def _group_by_filename(filenames):
    """Group files by stripping augmentation suffixes. Returns {group_key: [filenames]}."""
    groups = defaultdict(list)
    for fname in filenames:
        key = _strip_aug_suffix(fname)
        groups[key].append(fname)
    return dict(groups)


def _load_clip_model(device):
    """Load OpenCLIP model for embedding computation."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def _compute_clip_embeddings(image_paths, model, preprocess, device, batch_size=64):
    """Compute normalized CLIP embeddings for a list of image paths."""
    import open_clip
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(preprocess(img))
        batch = torch.stack(images).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _merge_groups_by_clip(groups, class_dir, model, preprocess, device, threshold=0.95):
    """
    For groups that have only 1 member (no filename-based match),
    compute CLIP embeddings and merge those with cosine similarity > threshold.
    """
    # Separate singleton groups from multi-member groups
    singletons = {}
    multi = {}
    for key, files in groups.items():
        if len(files) == 1:
            singletons[key] = files
        else:
            multi[key] = files

    if len(singletons) <= 1:
        # Nothing to merge
        groups.update(multi)
        return groups

    singleton_keys = list(singletons.keys())
    singleton_paths = [os.path.join(class_dir, singletons[k][0]) for k in singleton_keys]

    # Compute CLIP embeddings
    embeddings = _compute_clip_embeddings(singleton_paths, model, preprocess, device)

    # Cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # Union-Find to merge similar singletons
    parent = list(range(len(singleton_keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(singleton_keys)):
        for j in range(i + 1, len(singleton_keys)):
            if sim_matrix[i, j] > threshold:
                union(i, j)

    # Build merged groups
    merged = defaultdict(list)
    for i, key in enumerate(singleton_keys):
        root = find(i)
        root_key = singleton_keys[root]
        merged[root_key].extend(singletons[key])

    # Combine with multi-member groups
    result = dict(multi)
    result.update(dict(merged))
    return result


def _select_groups(groups, target_count, seed):
    """
    Select groups until we reach target_count images.
    Prioritizes groups with exactly 1 member (original only) to maximize diversity.
    """
    rng = random.Random(seed)

    # Sort: singletons first (more unique leaves), then by group size ascending
    sorted_keys = sorted(groups.keys(), key=lambda k: (len(groups[k]) > 1, len(groups[k])))

    selected_files = []
    selected_groups = []
    for key in sorted_keys:
        if len(selected_files) >= target_count:
            break
        files = groups[key]
        remaining = target_count - len(selected_files)
        if len(files) <= remaining:
            selected_files.extend(files)
            selected_groups.append(key)
        else:
            # Take a subset from this group
            subset = rng.sample(files, remaining)
            selected_files.extend(subset)
            selected_groups.append(key)

    return selected_files, selected_groups


def prepare_dataset(
    source_train_dir="tomato/train",
    source_valid_dir="tomato/valid",
    dest_dir="datasets",
    train_per_class=1000,
    test_per_class=200,
    seed=42,
    clip_threshold=0.95,
    use_clip=True,
):
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, clip_preprocess = None, None
    if use_clip:
        print("Loading CLIP model...")
        clip_model, clip_preprocess = _load_clip_model(device)

    train_dest = os.path.join(dest_dir, "train")
    test_dest = os.path.join(dest_dir, "test")
    group_info = {}

    for class_name in CLASS_NAMES:
        print(f"\n{'='*50}")
        print(f"Processing class: {class_name}")
        print(f"{'='*50}")

        # --- TRAIN ---
        src_train = os.path.join(source_train_dir, class_name)
        dst_train = os.path.join(train_dest, class_name)
        _process_split(
            src_train, dst_train, train_per_class, class_name, "train",
            clip_model, clip_preprocess, device, clip_threshold, seed, group_info,
        )

        # --- TEST ---
        src_test = os.path.join(source_valid_dir, class_name)
        dst_test = os.path.join(test_dest, class_name)
        _process_split(
            src_test, dst_test, test_per_class, class_name, "test",
            clip_model, clip_preprocess, device, clip_threshold, seed, group_info,
        )

    # Save group info for k-fold
    group_path = os.path.join(dest_dir, "group_mapping.json")
    with open(group_path, "w") as f:
        json.dump(group_info, f, indent=2)
    print(f"\nGroup mapping saved to {group_path}")

    # Cleanup CLIP
    if clip_model is not None:
        del clip_model
        torch.cuda.empty_cache()

    _print_summary(dest_dir)


def _process_split(
    src_dir, dst_dir, target_count, class_name, split_name,
    clip_model, clip_preprocess, device, threshold, seed, group_info,
):
    if not os.path.isdir(src_dir):
        print(f"  WARNING: {src_dir} not found")
        return

    # Step 1: Filter 256x256
    print(f"  [{split_name}] Filtering 256x256 images...")
    valid_files = _filter_256(src_dir)
    print(f"  [{split_name}] {len(valid_files)} images are 256x256")

    # Step 2: Group by filename
    groups = _group_by_filename(valid_files)
    num_filename_groups = len(groups)

    # Step 3: CLIP-based merging
    if clip_model is not None:
        print(f"  [{split_name}] Computing CLIP embeddings for singleton groups...")
        groups = _merge_groups_by_clip(
            groups, src_dir, clip_model, clip_preprocess, device, threshold
        )

    num_final_groups = len(groups)
    total_images = sum(len(v) for v in groups.values())
    multi_groups = sum(1 for v in groups.values() if len(v) > 1)
    print(
        f"  [{split_name}] Groups: {num_filename_groups} (filename) -> {num_final_groups} (after CLIP merge)"
    )
    print(f"  [{split_name}] {multi_groups} groups have multiple images (augmentations)")

    # Step 4: Select
    selected_files, selected_group_keys = _select_groups(groups, target_count, seed)
    print(f"  [{split_name}] Selected {len(selected_files)} images from {len(selected_group_keys)} groups")

    # Step 5: Copy
    os.makedirs(dst_dir, exist_ok=True)
    for fname in selected_files:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    # Step 6: Save group mapping (for k-fold)
    file_to_group = {}
    for gkey in selected_group_keys:
        for fname in groups[gkey]:
            if fname in selected_files:
                file_to_group[fname] = gkey
    info_key = f"{split_name}/{class_name}"
    group_info[info_key] = file_to_group


def _print_summary(dest_dir):
    print(f"\n{'='*50}")
    print("Dataset Summary")
    print(f"{'='*50}")
    for split in ["train", "test"]:
        split_dir = os.path.join(dest_dir, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"\n  {split}/")
        total = 0
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                n = len(os.listdir(cls_dir))
                total += n
                print(f"    {cls}: {n}")
        print(f"    TOTAL: {total}")
