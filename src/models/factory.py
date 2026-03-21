import timm


def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Create a timm model. model_name is passed directly as the timm model ID."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
