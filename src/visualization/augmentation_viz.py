import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from src.data.transforms import get_train_transforms
from src.constants.labels import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
import numpy as np
import torch


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_augmentations(dataset_path, image_size, num_augmented=5, seed=42, save_path=None):
    random.seed(seed)
    transform = get_train_transforms(image_size)
    train_dir = os.path.join(dataset_path, "train")

    num_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(num_classes, num_augmented + 1, figsize=(3 * (num_augmented + 1), 3 * num_classes))

    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        if not files:
            continue

        img_path = os.path.join(class_dir, random.choice(files))
        img = Image.open(img_path).convert("RGB")

        # Original
        axes[row][0].imshow(img.resize((image_size, image_size)))
        axes[row][0].set_title(f"{class_name}\n(original)", fontsize=7)
        axes[row][0].axis("off")

        # Augmented versions
        for col in range(1, num_augmented + 1):
            aug_tensor = transform(img)
            aug_img = denormalize(aug_tensor)
            axes[row][col].imshow(aug_img)
            axes[row][col].set_title(f"Aug {col}", fontsize=7)
            axes[row][col].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved augmentation visualization to {save_path}")
    plt.close()
