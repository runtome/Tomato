# Tomato Leaf Disease Classification

CNN-based classification of tomato leaf diseases using pretrained models from [timm](https://github.com/huggingface/pytorch-image-models). Compares 5 architectures via 5-fold stratified cross-validation for research evaluation, with the goal of deploying a lightweight model on mobile.

## Classes (10)

| # | Class Name |
|---|-----------|
| 0 | Bacterial_spot |
| 1 | Early_blight |
| 2 | Late_blight |
| 3 | Leaf_Mold |
| 4 | Septoria_leaf_spot |
| 5 | Spider_mites Two-spotted_spider_mite |
| 6 | Target_Spot |
| 7 | Tomato_Yellow_Leaf_Curl_Virus |
| 8 | Tomato_mosaic_virus |
| 9 | healthy |

## Models

| Model | timm ID | Image Size |
|-------|---------|-----------|
| MobileNetV3 | `mobilenetv3_large_100.ra_in1k` | 224 |
| MobileNetV4 | `mobilenetv4_conv_medium.e500_r224_in1k` | 224 |
| InceptionV3 | `inception_v3.tv_in1k` | 299 |
| EfficientNetB1 | `efficientnet_b1.ra_in1k` | 224 |
| ResNet50 | `resnet50.a1_in1k` | 224 |

## Project Structure

```
Tomato/
├── src/
│   ├── constants/labels.py          # Class names, model registry, ImageNet stats
│   ├── configs/config.py            # Config dataclass + YAML loader
│   ├── configs/*.yaml               # Per-model config files
│   ├── models/factory.py            # timm model creation
│   ├── data/
│   │   ├── dataset.py               # TomatoDataset class
│   │   ├── transforms.py            # Train/val augmentations
│   │   ├── kfold.py                 # StratifiedGroupKFold (augmentation-aware)
│   │   └── prepare_dataset.py       # CLIP-based dedup + image selection
│   ├── utils/                       # seed, device, checkpoint, early stopping, timer
│   ├── metrics/evaluation.py        # Accuracy, precision, recall, F1, t-test
│   ├── visualization/
│   │   ├── plots.py                 # Loss curves, confusion matrix heatmaps
│   │   ├── gradcam.py               # Grad-CAM generation
│   │   └── augmentation_viz.py      # Augmentation comparison grid
│   └── pipelines/
│       ├── trainer.py               # Training loop with mixed precision
│       └── evaluator.py             # Evaluation + metrics aggregation
├── scripts/                         # CLI entry points
├── datasets/                        # Created by prepare_dataset
│   ├── train/{class}/               # 1000 images/class
│   ├── test/{class}/                # 200 images/class
│   └── group_mapping.json           # Image-to-group mapping for k-fold
└── outputs/models/{save_name}/      # Checkpoints, graphs, Grad-CAM
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Selects 256x256 images from `tomato/` into `datasets/`, with CLIP-based deduplication to prevent data leakage during k-fold cross-validation.

**Problem:** The source dataset contains pre-augmented images (flipped, rotated copies of the same leaf). If these end up in different folds, the model sees near-identical images in both train and validation, inflating metrics.

**Solution pipeline:**
1. **Filter** — only 256x256 images are selected (other sizes are excluded)
2. **Filename grouping** — strips augmentation suffixes (`_flipLR`, `_flipTB`, `_180deg`, `_90deg`, `_270deg`, `_new30degFlipLR`) to group images from the same original leaf
3. **CLIP merging** — for images without filename-based matches, computes CLIP embeddings (ViT-B-32) and merges groups with cosine similarity > 0.95 to catch near-duplicates with different naming conventions
4. **Group-level selection** — picks groups (not individual images) until reaching the target count, prioritizing singleton groups for maximum leaf diversity
5. **Saves `group_mapping.json`** — maps each selected image to its group ID, used by `StratifiedGroupKFold` to keep all augmented variants of the same leaf in the same fold

```bash
# Full pipeline with CLIP deduplication (recommended)
python scripts/prepare_dataset.py

# Filename-based grouping only (faster, no CLIP model download)
python scripts/prepare_dataset.py --no-clip

# Custom similarity threshold
python scripts/prepare_dataset.py --clip-threshold 0.90
```

| Split | Source | Images/class | Total |
|-------|--------|-------------|-------|
| Train | `tomato/train/` | 1,000 | 10,000 |
| Test  | `tomato/valid/` | 200   | 2,000  |

### 2. Show Model Summary

```bash
python scripts/show_summary.py --config src/configs/mobilenetv3.yaml
```

### 3. Train

Train a single fold:

```bash
python scripts/train.py --config src/configs/mobilenetv3.yaml --fold 0
```

Train all 5 folds:

```bash
python scripts/train.py --config src/configs/mobilenetv3.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py --config src/configs/mobilenetv3.yaml --fold 0
```

Evaluate all folds (aggregated mean ± SD + t-test ready):

```bash
python scripts/evaluate.py --config src/configs/mobilenetv3.yaml
```

### 5. Grad-CAM

```bash
python scripts/gradcam.py --config src/configs/mobilenetv3.yaml --fold 0 --num_images 3
```

### 6. Visualize Augmentations

```bash
python scripts/visualize_augmentation.py --config src/configs/mobilenetv3.yaml
```

## Training Details

- **Cross-validation:** 5-fold StratifiedGroupKFold — augmentation-aware splits prevent data leakage (same seed across models for fair comparison)
- **Augmentations:** RandomResizedCrop, horizontal/vertical flip, rotation (±45°)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR
- **Mixed precision:** Enabled by default (AMP)
- **Early stopping:** Patience = 5 epochs

## Outputs

```
outputs/models/{save_name}/
├── {save_name}_fold_{N}.pth              # Best checkpoint per fold
├── train_graph/{save_name}_fold_{N}.png  # Loss curves
├── confusion-matrix/{save_name}_fold_{N}.png
└── Grad-CAM/{class_name}_sample{N}.jpg
```
