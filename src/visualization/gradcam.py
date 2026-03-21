import os
import numpy as np
import cv2
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.data.transforms import get_val_transforms
from src.constants.labels import IMAGENET_MEAN, IMAGENET_STD


TARGET_LAYER_MAP = {
    "MobileNetV3": lambda m: m.blocks[-1],
    "MobileNetV4": lambda m: m.blocks[-1],
    "InceptionV3": lambda m: m.Mixed_7c,
    "EfficientNetB1": lambda m: m.blocks[-1],
    "ResNet50": lambda m: m.layer4[-1],
}


def get_target_layer(model, model_name):
    if model_name not in TARGET_LAYER_MAP:
        raise ValueError(f"No target layer defined for '{model_name}'")
    return [TARGET_LAYER_MAP[model_name](model)]


def generate_gradcam(model, model_name, image_path, image_size, device):
    target_layers = get_target_layer(model, model_name)

    transform = get_val_transforms(image_size)
    img_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Original image normalized to [0, 1] for overlay
    img_resized = img_pil.resize((image_size, image_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0]

    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return img_np, overlay


def save_gradcam_grid(original, overlay, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
