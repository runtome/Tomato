CLASS_NAMES = sorted([
    "Bacterial_spot",
    "Early_blight",
    "healthy",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus",
])

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

MODEL_REGISTRY = {
    "MobileNetV3": "mobilenetv3_large_100.ra_in1k",
    "MobileNetV4": "mobilenetv4_conv_medium.e500_r224_in1k",
    "InceptionV3": "inception_v3.tv_in1k",
    "EfficientNetB1": "efficientnet_b1.ra_in1k",
    "ResNet50": "resnet50.a1_in1k",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
