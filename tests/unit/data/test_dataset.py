import os
import pytest
import pandas as pd
import torch
import shutil
import tempfile
from PIL import Image
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized

# Add the project's root directory to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.data.dataset import CXRFractureDataset

@pytest.fixture(scope="module")
def fake_data_environment():
    """
    A pytest fixture to create a temporary, fake dataset environment for all tests in this module.
    Yields the paths to the CSV and image root directory.
    """
    test_dir = tempfile.mkdtemp()
    image_root_dir = os.path.join(test_dir, "files")
    csv_dir = os.path.join(test_dir, "splits", "test_split")

    os.makedirs(os.path.join(image_root_dir, "p10/p10000032/s50414267"), exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    img1_path = os.path.join(image_root_dir, "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg")
    Image.new('L', (512, 512)).save(img1_path)
    
    csv_path = os.path.join(csv_dir, "test.csv")
    data = {
        'dicom_id': ['02aa804e-bde0afdd-112c0b34-7bc16630-4e384014'],
        'study_id': [50414267],
        'subject_id': [10000032],
        'fracture': [1]
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    yield {"csv_path": csv_path, "image_root_dir": image_root_dir}
    
    shutil.rmtree(test_dir)

def test_initialization_and_len(fake_data_environment):
    """Test if the dataset initializes correctly and __len__ works."""
    dataset = CXRFractureDataset(
        csv_path=fake_data_environment["csv_path"], 
        image_root_dir=fake_data_environment["image_root_dir"]
    )
    assert len(dataset) == 1

def test_getitem_content(fake_data_environment):
    """Test if __getitem__ returns a path string without transforms."""
    dataset = CXRFractureDataset(
        csv_path=fake_data_environment["csv_path"], 
        image_root_dir=fake_data_environment["image_root_dir"]
    )
    
    sample = dataset[0]
    # *** FIX: Assert that the returned 'image' is a string path ***
    assert isinstance(sample['image'], str)
    assert os.path.exists(sample['image']) # Check that the path is valid
    assert sample['label'] == 1

def test_getitem_with_transform(fake_data_environment):
    """Test if __getitem__ applies MONAI transforms correctly."""
    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128)),
    ])
    dataset = CXRFractureDataset(
        csv_path=fake_data_environment["csv_path"], 
        image_root_dir=fake_data_environment["image_root_dir"],
        transform=transform
    )
    
    sample = dataset[0]
    image, label = sample['image'], sample['label']

    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 128, 128)
    assert isinstance(label, torch.Tensor)
    assert label == 1

def test_file_not_found_error_csv(fake_data_environment):
    """Test if FileNotFoundError is raised for a missing CSV file."""
    with pytest.raises(FileNotFoundError):
        CXRFractureDataset(csv_path="non_existent_file.csv", image_root_dir=fake_data_environment["image_root_dir"])