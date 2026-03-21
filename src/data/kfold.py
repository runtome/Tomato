import os
import json
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from src.constants.labels import CLASS_TO_IDX


def collect_all_samples(root_dir, class_to_idx=None):
    """Scan directory and return list of (path, label) tuples."""
    if class_to_idx is None:
        class_to_idx = CLASS_TO_IDX
    samples = []
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath):
                samples.append((fpath, idx))
    return samples


def _load_group_mapping(dataset_path):
    """Load group_mapping.json produced by prepare_dataset."""
    group_path = os.path.join(dataset_path, "group_mapping.json")
    if not os.path.exists(group_path):
        return None
    with open(group_path, "r") as f:
        return json.load(f)


def _get_groups_for_samples(samples, group_mapping):
    """
    Map each sample to its group ID using the group mapping.
    Returns a list of group IDs parallel to samples.
    """
    groups = []
    group_id_map = {}
    next_id = 0

    for path, label in samples:
        # Extract class_name and filename from path
        parts = path.replace("\\", "/").split("/")
        class_name = parts[-2]
        fname = parts[-1]

        key = f"train/{class_name}"
        group_key = None
        if group_mapping and key in group_mapping:
            group_key = group_mapping[key].get(fname)

        if group_key is None:
            # No group info — treat as its own group
            group_key = f"_solo_{fname}"

        if group_key not in group_id_map:
            group_id_map[group_key] = next_id
            next_id += 1
        groups.append(group_id_map[group_key])

    return groups


def create_kfold_splits(samples, num_folds, seed, dataset_path=None):
    """
    Create stratified k-fold splits.
    If group_mapping.json exists in dataset_path, uses StratifiedGroupKFold
    to keep augmented variants of the same leaf in the same fold.
    Otherwise falls back to StratifiedKFold.
    """
    labels = [label for _, label in samples]
    dummy_X = list(range(len(samples)))

    group_mapping = None
    if dataset_path:
        group_mapping = _load_group_mapping(dataset_path)

    if group_mapping:
        groups = _get_groups_for_samples(samples, group_mapping)
        skf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        iterator = skf.split(dummy_X, labels, groups=groups)
        print(f"Using StratifiedGroupKFold ({num_folds} folds) — augmentation-aware splits")
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        iterator = skf.split(dummy_X, labels)
        print(f"Using StratifiedKFold ({num_folds} folds) — no group mapping found")

    splits = []
    for train_idx, val_idx in iterator:
        splits.append((train_idx.tolist(), val_idx.tolist()))
    return splits
