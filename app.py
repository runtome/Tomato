import glob
import os

import streamlit as st
import torch
from PIL import Image

from src.configs.config import Config
from src.constants.labels import IDX_TO_CLASS, NUM_CLASSES
from src.data.transforms import get_val_transforms
from src.models.factory import create_model
from src.utils.checkpoint import load_checkpoint

FRIENDLY_LABELS = {
    "Bacterial_spot": "Bacterial Spot",
    "Early_blight": "Early Blight",
    "healthy": "Healthy",
    "Late_blight": "Late Blight",
    "Leaf_Mold": "Leaf Mold",
    "Septoria_leaf_spot": "Septoria Leaf Spot",
    "Spider_mites Two-spotted_spider_mite": "Spider Mites (Two-spotted)",
    "Target_Spot": "Target Spot",
    "Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
}

MODELS_DIR = "models"
IMAGES_DIR = "images"
CONFIGS_DIR = os.path.join("src", "configs")


def discover_models():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def discover_example_images():
    paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    paths += sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


@st.cache_resource
def load_model(model_name, weights_path):
    config_path = os.path.join(CONFIGS_DIR, f"{model_name}.yaml")
    if not os.path.exists(config_path):
        return None, None, f"Config not found: {config_path}"

    config = Config.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config.model_name, config.num_classes, pretrained=False)

    try:
        load_checkpoint(model, weights_path, device=device)
    except Exception as e:
        return None, None, f"Failed to load weights: {e}"

    model.to(device)
    model.eval()
    return model, config, None


def predict(model, image, image_size, device):
    transform = get_val_transforms(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    all_probs = {}
    for i in range(NUM_CLASSES):
        class_name = IDX_TO_CLASS[i]
        all_probs[class_name] = probs[i].item()

    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top_class, top_prob = sorted_probs[0]

    return top_class, top_prob, sorted_probs


def main():
    st.set_page_config(
        page_title="Tomato Leaf Disease Classifier",
        page_icon="🍅",
        layout="centered",
    )

    st.title("Tomato Leaf Disease Classifier")

    # --- Sidebar: Model Selection ---
    st.sidebar.header("Model")
    models = discover_models()

    if not models:
        st.sidebar.error(f"No .pth files found in `{MODELS_DIR}/`")
        st.stop()

    selected_name = st.sidebar.selectbox("Select model", list(models.keys()))
    weights_path = models[selected_name]

    model, config, error = load_model(selected_name, weights_path)
    if error:
        st.sidebar.error(error)
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.success(f"Loaded **{selected_name}** on `{device}`")

    # --- Main: Image Input ---
    st.subheader("Upload or select an image")

    uploaded_file = st.file_uploader(
        "Upload a tomato leaf image", type=["jpg", "jpeg", "png"]
    )

    example_images = discover_example_images()
    example_choice = None
    if example_images:
        example_choice = st.selectbox(
            "Or pick an example image",
            ["(none)"] + list(example_images.keys()),
        )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif example_choice and example_choice != "(none)":
        image = Image.open(example_images[example_choice])

    if image is None:
        st.info("Please upload an image or select an example above.")
        st.stop()

    # --- Prediction ---
    with st.spinner("Predicting..."):
        top_class, top_prob, sorted_probs = predict(
            model, image, config.image_size, device
        )

    friendly_top = FRIENDLY_LABELS.get(top_class, top_class)

    if top_class == "healthy":
        st.success(f"**{friendly_top}** — {top_prob:.1%} confidence")
    else:
        st.warning(f"**{friendly_top}** — {top_prob:.1%} confidence")

    # --- Image (left) + Probabilities (right) ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.image(image, caption="Input Image", use_container_width=True)

    with col_right:
        st.subheader("All class probabilities")
        for class_name, prob in sorted_probs:
            friendly = FRIENDLY_LABELS.get(class_name, class_name)
            st.text(f"{friendly:<30s} {prob:>7.2%}")


if __name__ == "__main__":
    main()
