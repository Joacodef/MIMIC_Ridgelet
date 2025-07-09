import pytest
import torch
import torch.nn as nn
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

# Adjust the import path based on your project structure
from src.training.train import train_one_epoch, validate, run_training
from src.config.config import (
    AppConfig, 
    DataConfig, 
    ModelConfig, 
    TrainingConfig, 
    WandbConfig, 
    AugmentationConfig,
    TransformParams,
    WaveletTransformConfig, 
    LossConfig, 
    SchedulerConfig,
    DataLoaderConfig
)

# --- Test Fixtures ---

@pytest.fixture
def mock_loader():
    """Creates a mock DataLoader that yields a single batch of dummy data."""
    # Batch of 4 images, 1 channel, 32x32 pixels
    images = torch.randn(4, 1, 32, 32)
    # Corresponding labels
    labels = torch.randint(0, 2, (4,))
    # The dataloader will yield this single batch
    return [_ for _ in [{"image": images, "label": labels}]]

@pytest.fixture
def mock_model():
    """Creates a simple mock model."""
    # A simple model that can process the mock data's shape
    return nn.Sequential(nn.Conv2d(1, 1, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 1))

@pytest.fixture
def basic_config(tmp_path: Path) -> AppConfig:
    """Creates a minimal but valid AppConfig for testing the run_training function."""
    
    # Define and create a temporary directory for the cache
    cache_directory = tmp_path / "persistent_cache"
    cache_directory.mkdir()

    return AppConfig(
        pathology="test_pathology",
        data=DataConfig(
            train_size=100,
            image_size=32,
            # This is the fix 👇
            persistent_cache_dir=str(cache_directory),
            transform_name='none',
            augmentations=AugmentationConfig(),
            transform_params=TransformParams(
                wavelet=WaveletTransformConfig()
            )
        ),
        model=ModelConfig(base_model='resnet18'),
        dataloader=DataLoaderConfig(batch_size=2, num_workers=1),
        training=TrainingConfig(
            epochs=1,
            learning_rate=1e-3,
            early_stopping_patience=3,
            loss=LossConfig(name='bcewithlogitsloss'),
            scheduler=SchedulerConfig(name='none')
        ),
        wandb=WandbConfig(enabled=False, project="test_project", entity="test_entity")
    )
# --- Tests for Core Functions ---

def test_train_one_epoch_updates_weights(mock_loader, mock_model):
    """
    1. Most Important Test: Verifies the core training logic.
    
    This test ensures that after one training step, the model's weights have
    been updated, which confirms that gradients were calculated and applied.
    """
    # Arrange
    device = torch.device("cpu")
    model = mock_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Capture the initial state of a weight parameter
    initial_weight = model[0].weight.clone()

    # Act
    train_loss = train_one_epoch(model, mock_loader, optimizer, criterion, device)

    # Assert
    final_weight = model[0].weight
    
    assert isinstance(train_loss, float)
    # The core assertion: the weights must have changed after the optimizer step
    assert not torch.equal(initial_weight, final_weight)

def test_validate_calculates_auc_correctly():
    """
    2. Most Important Test: Verifies the validation logic.
    
    This test ensures that the validation function correctly computes metrics
    like AUC by providing predictable inputs and checking for a known output.
    """
    # Arrange
    device = torch.device("cpu")
    model = MagicMock()
    criterion = nn.BCEWithLogitsLoss()

    # Create predictable outputs and labels
    # Model outputs logits that correspond to probabilities [0.1, 0.9, 0.2, 0.8]
    # This ordering should result in a perfect AUC score of 1.0
    labels = torch.tensor([0, 1, 0, 1])
    logits = torch.tensor([-2.2, 2.2, -1.4, 1.4]).unsqueeze(1) # Inverse of sigmoid
    
    # Mock the model to return these specific logits
    model.return_value = logits
    
    # Create a mock loader that yields this predictable data
    val_loader = [{"image": torch.randn(4, 1, 32, 32), "label": labels}]

    # Act
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    # Assert
    assert isinstance(val_loss, float)
    assert val_auc == 1.0

# --- Integration "Smoke Test" ---

@patch('src.training.train.validate', return_value=(0.5, 0.95))
@patch('src.training.train.train_one_epoch', return_value=0.8)
@patch('src.training.train.wandb')
@patch('src.training.train.os.getenv')
def test_run_training_smoke_test(mock_getenv, mock_wandb, mock_train_one_epoch, mock_validate, basic_config, tmp_path):
    """
    3. Most Important Test: A "smoke test" for the main orchestrator.
    """
    # Arrange: Mock filesystem and environment variables
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    test_run_name = "test_run_smoke"
    output_run_dir = output_dir / "models" / basic_config.pathology / test_run_name

    # --- FIX: Update the lambda to accept a 'default' argument ---
    # This matches the real os.getenv(key, default=None) signature.
    mock_getenv.side_effect = lambda key, default=None: str(data_dir) if 'DATA' in key else str(output_dir)

    # Create dummy data files that CXRClassificationDataset will look for
    split_dir = data_dir / "splits" / f"split_{basic_config.pathology}_{basic_config.data.train_size}"
    split_dir.mkdir(parents=True)
    (split_dir / "train.csv").write_text("dicom_id,study_id,subject_id,label\nfake1,1,1,1")
    (split_dir / "validation.csv").write_text("dicom_id,study_id,subject_id,label\nfake2,2,2,0")

    # Act: Run the main training function
    best_auc, final_run_dir = run_training(basic_config, output_dir_override=str(output_run_dir))

    # Assert
    mock_train_one_epoch.assert_called_once()
    mock_validate.assert_called_once()
    assert best_auc == 0.95
    assert final_run_dir == str(output_run_dir)
    assert (Path(final_run_dir) / "config.yaml").is_file()
    assert (Path(final_run_dir) / "best_model.pth").is_file()
    assert (Path(final_run_dir) / "last_model.pth").is_file()