import timm
from src.constants.labels import MODEL_REGISTRY


def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    timm_id = MODEL_REGISTRY[model_name]
    model = timm.create_model(timm_id, pretrained=pretrained, num_classes=num_classes)
    return model
