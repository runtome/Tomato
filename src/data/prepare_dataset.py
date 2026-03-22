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


def _merge_groups_by_clip(groups, class_dir, model, preprocess, device, threshold=0.98):
    """
    For groups that have only 1 member (no filename-based match),
    compute CLIP embeddings and merge near-duplicates.

    Uses clique-based merging instead of Union-Find to prevent transitive chaining:
    all members in a merged group must have pairwise similarity > threshold.
    """
    singletons = {}
    multi = {}
    for key, files in groups.items():
        if len(files) == 1:
            singletons[key] = files
        else:
            multi[key] = files

    if len(singletons) <= 1:
        result = dict(multi)
        result.update(singletons)
        return result

    singleton_keys = list(singletons.keys())
    singleton_paths = [os.path.join(class_dir, singletons[k][0]) for k in singleton_keys]

    embeddings = _compute_clip_embeddings(singleton_paths, model, preprocess, device)
    sim_matrix = embeddings @ embeddings.T

    # Greedy clique-based merging:
    # For each image, find its nearest neighbor. If sim > threshold AND the neighbor
    # also has this image as a top match, merge them. Then grow the cluster only if
    # a new candidate has sim > threshold with ALL existing members.
    n = len(singleton_keys)
    assigned = [False] * n
    merged = {}

    # Sort pairs by similarity (highest first), only consider pairs above threshold
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > threshold:
                pairs.append((sim_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    # Build clusters: each cluster requires all pairwise sims > threshold
    clusters = []  # list of sets
    idx_to_cluster = {}

    for sim, i, j in pairs:
        ci = idx_to_cluster.get(i)
        cj = idx_to_cluster.get(j)

        if ci is not None and cj is not None:
            # Both assigned — try to merge clusters if all cross-pairs above threshold
            if ci == cj:
                continue
            cluster_i = clusters[ci]
            cluster_j = clusters[cj]
            can_merge = True
            for a in cluster_i:
                for b in cluster_j:
                    if sim_matrix[a, b] <= threshold:
                        can_merge = False
                        break
                if not can_merge:
                    break
            if can_merge:
                # Merge j into i
                for idx in cluster_j:
                    cluster_i.add(idx)
                    idx_to_cluster[idx] = ci
                clusters[cj] = set()  # empty the old cluster
        elif ci is not None:
            # i assigned, check if j fits in i's cluster
            cluster = clusters[ci]
            if all(sim_matrix[j, m] > threshold for m in cluster):
                cluster.add(j)
                idx_to_cluster[j] = ci
        elif cj is not None:
            # j assigned, check if i fits in j's cluster
            cluster = clusters[cj]
            if all(sim_matrix[i, m] > threshold for m in cluster):
                cluster.add(i)
                idx_to_cluster[i] = cj
        else:
            # Neither assigned — new cluster
            new_idx = len(clusters)
            clusters.append({i, j})
            idx_to_cluster[i] = new_idx
            idx_to_cluster[j] = new_idx

    # Build result
    result = dict(multi)

    # Add clusters
    used = set()
    for cluster in clusters:
        if not cluster:
            continue
        members = sorted(cluster)
        root_key = singleton_keys[members[0]]
        for m in members:
            result.setdefault(root_key, []).extend(singletons[singleton_keys[m]])
            used.add(m)

    # Add remaining singletons (not merged)
    for i, key in enumerate(singleton_keys):
        if i not in used:
            result[key] = singletons[key]

    # Print similarity diagnostics
    if pairs:
        top_sims = [s for s, _, _ in pairs[:10]]
        print(f"    CLIP: {len(pairs)} pairs above threshold {threshold:.2f}, "
              f"top similarities: {', '.join(f'{s:.4f}' for s in top_sims)}")
    else:
        print(f"    CLIP: no pairs above threshold {threshold:.2f}")

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
    clip_threshold=0.98,
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

    # Step 0: Clean destination directory
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)

    # Step 1: Filter 256x256
    total_images = len([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])
    print(f"  [{split_name}] Total images in folder: {total_images}")
    print(f"  [{split_name}] Filtering 256x256 images...")
    valid_files = _filter_256(src_dir)
    print(f"  [{split_name}] {len(valid_files)} images are 256x256")

    # Step 2: Group by filename
    groups = _group_by_filename(valid_files)
    num_filename_groups = len(groups)
    num_filename_multi = sum(1 for v in groups.values() if len(v) > 1)

    # Step 3: CLIP-based merging
    if clip_model is not None:
        num_singletons = sum(1 for v in groups.values() if len(v) == 1)
        print(f"  [{split_name}] Computing CLIP embeddings for {num_singletons} singleton groups...")
        groups = _merge_groups_by_clip(
            groups, src_dir, clip_model, clip_preprocess, device, threshold
        )

    num_final_groups = len(groups)
    multi_groups = sum(1 for v in groups.values() if len(v) > 1)

    # Print group size distribution
    sizes = [len(v) for v in groups.values()]
    size_dist = defaultdict(int)
    for s in sizes:
        size_dist[s] += 1
    dist_str = ", ".join(f"size {k}: {v}" for k, v in sorted(size_dist.items()))

    print(f"  [{split_name}] Groups: {num_filename_groups} (filename, {num_filename_multi} multi)"
          f" -> {num_final_groups} (after CLIP)")
    print(f"  [{split_name}] {multi_groups} groups have augmentations | Distribution: {dist_str}")

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
