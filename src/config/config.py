import yaml
from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Any, Optional, List

# --- Configuration Data Classes ---

@dataclass
class AugmentationConfig:
    """Configuration for data augmentations."""
    rand_flip_prob: float = 0.5
    rand_affine_prob: float = 0.5
    rotate_range: float = 0.1
    scale_range: float = 0.1

@dataclass
class HaarTransformConfig:
    """Configuration specific to the Haar Transform."""
    levels: int = 1
    details_to_keep: List[str] = field(default_factory=lambda: ["HL", "VL", "DL"])

@dataclass
class TransformParams:
    """Container for specific transform parameters."""
    haar: HaarTransformConfig = field(default_factory=HaarTransformConfig)

@dataclass
class DataConfig:
    split_folder_name: str
    image_size: int = 256
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    transform_name: Optional[str] = None
    transform_threshold_ratio: float = 0.1
    multichannel: bool = False
    transform_params: TransformParams = field(default_factory=TransformParams)

@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class LossConfig:
    """Configuration for the loss function."""
    name: str = "FocalLoss" # Options: "FocalLoss", "BCEWithLogitsLoss"
    gamma: float = 2.0 # The gamma parameter for FocalLoss
    alpha: Optional[float] = None

@dataclass
class ModelConfig:
    base_model: str = "resnet18"

@dataclass
class SchedulerConfig:
    """Configuration for the learning rate scheduler."""
    name: str = "ReduceLROnPlateau"
    # Factor by which the learning rate will be reduced. new_lr = lr * factor.
    factor: float = 0.1
    # Number of epochs with no improvement after which learning rate will be reduced.
    patience: int = 5

@dataclass
class TrainingConfig:
    optimizer: str = "Adam"
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    output_model_name: str = "best_model.pth"
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

@dataclass
class WandbConfig:
    """Configuration for Weights & Biases integration."""
    enabled: bool
    project: str
    entity: Optional[str]

@dataclass
class AppConfig:
    data: DataConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig

# --- Configuration Loader ---

def _from_dict(data_class: Any, data: dict) -> Any:
    """
    Recursively creates a dataclass instance from a dictionary.
    """
    if not is_dataclass(data_class):
        return data

    field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
    
    init_args = {}
    for key, value in data.items():
        if key in field_types:
            # Recursively call _from_dict for nested dataclasses
            init_args[key] = _from_dict(field_types[key], value)
    
    return data_class(**init_args)


def load_config(path: str) -> AppConfig:
    """
    Loads configuration from a YAML file, sets defaults, and returns a
    structured AppConfig object.
    """
    with open(path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Create the final configuration object using the recursive helper
    config = _from_dict(AppConfig, yaml_data)
    
    return config