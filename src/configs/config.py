from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class Config:
    model_name: str = "MobileNetV3"
    save_name: str = "mobilenetv3"
    num_classes: int = 10
    epochs: int = 30
    lr: float = 1e-3
    optimizer: str = "Adam"
    weight_decay: float = 1e-4
    batch_size: int = 32
    image_size: int = 224
    num_workers: int = 4
    num_folds: int = 5
    fold: Optional[int] = None
    random_seed: int = 42
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    scheduler: str = "CosineAnnealingLR"
    scheduler_params: dict = field(default_factory=lambda: {"T_max": 30, "eta_min": 1e-6})
    dataset_path: str = "datasets"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
