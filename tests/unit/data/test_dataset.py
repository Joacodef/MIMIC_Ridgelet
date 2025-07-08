import os
import pytest
import pandas as pd
import torch
import shutil
import tempfile
from PIL import Image
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized

# By running pytest from the project root, it will automatically find the `src` module.
# This avoids manipulating sys.path, which is a better practice.
from src.data.dataset import CXRClassificationDataset


@pytest.fixture(scope="module")
def fake_data_environment():
    """
    A pytest fixture to create a temporary, fake dataset environment.

    This fixture sets up a directory structure with two mock images and a corresponding
    CSV file. It yields the paths to the test environment and cleans up afterwards.
    """
    test_dir = tempfile.mkdtemp()
    image_root_dir = os.path.join(test_dir, "files")
    csv_path = os.path.join(test_dir, "test_split.csv")

    # --- Create two data points to make tests more robust ---
    # Image 1
    p_dir_1 = os.path.join(image_root_dir, "p10/p10000032/s50414267")
    os.makedirs(p_dir_1, exist_ok=True)
    img1_path = os.path.join(p_dir_1, "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg")
    Image.new('L', (512, 512)).save(img1_path)

    # Image 2
    p_dir_2 = os.path.join(image_root_dir, "p12/p12345678/s59876543")
    os.makedirs(p_dir_2, exist_ok=True)
    img2_path = os.path.join(p_dir_2, "3a3b5c6d-7e8f9a0b-1c2d3e4f-5g6h7i8j-9k0l1m2n.jpg")
    Image.new('L', (256, 256)).save(img2_path)
    
    # --- Changed 'fracture' to 'label' to match dataset implementation ---
    data = {
        'dicom_id': ['02aa804e-bde0afdd-112c0b34-7bc16630-4e384014', '3a3b5c6d-7e8f9a0b-1c2d3e4f-5g6h7i8j-9k0l1m2n'],
        'study_id': [50414267, 59876543],
        'subject_id': [10000032, 12345678],
        'label': [1, 0] 
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    yield {"csv_path": csv_path, "image_root_dir": image_root_dir}
    
    shutil.rmtree(test_dir)


# --- Grouped tests in a class for better organization ---
class TestCXRClassificationDataset:

    def test_initialization_and_len(self, fake_data_environment):
        """Test if the dataset initializes correctly and __len__ works."""
        dataset = CXRClassificationDataset(
            csv_path=fake_data_environment["csv_path"], 
            image_root_dir=fake_data_environment["image_root_dir"]
        )
        assert len(dataset) == 2

    def test_getitem_content(self, fake_data_environment):
        """Test if __getitem__ returns a path and correct label without transforms."""
        dataset = CXRClassificationDataset(
            csv_path=fake_data_environment["csv_path"],
            image_root_dir=fake_data_environment["image_root_dir"]
        )
        
        sample = dataset[0]
        
        # This ensures the correct path separators are used on any OS.
        expected_path = os.path.join(
            fake_data_environment["image_root_dir"], 
            "p10", 
            "p10000032", 
            "s50414267", 
            "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
        )
        
        assert isinstance(sample['image'], str)
        assert sample['image'] == expected_path
        assert torch.equal(sample['label'], torch.tensor(1.0, dtype=torch.float32))

    def test_getitem_with_transform(self, fake_data_environment):
        """Test if __getitem__ applies MONAI transforms correctly."""
        transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(128, 128)),
        ])
        dataset = CXRClassificationDataset(
            csv_path=fake_data_environment["csv_path"], 
            image_root_dir=fake_data_environment["image_root_dir"],
            transform=transform
        )
        
        sample = dataset[1] # Test the second sample
        image, label = sample['image'], sample['label']

        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 128, 128)
        assert isinstance(label, torch.Tensor)
        assert torch.equal(label, torch.tensor(0.0, dtype=torch.float32))

    def test_csv_not_found_error(self, fake_data_environment):
        """Test if FileNotFoundError is raised for a missing CSV file."""
        with pytest.raises(FileNotFoundError, match="The specified CSV file was not found"):
            CXRClassificationDataset(csv_path="non_existent_file.csv", image_root_dir=fake_data_environment["image_root_dir"])

    def test_image_not_found_error(self, fake_data_environment):
        """Test for an error when the image file is missing from the disk."""
        transform = Compose([LoadImaged(keys=["image"])])
        dataset = CXRClassificationDataset(
            csv_path=fake_data_environment["csv_path"], 
            image_root_dir=fake_data_environment["image_root_dir"],
            transform=transform
        )
        
        # Tamper with the data to point to a non-existent file
        dataset.data_frame.loc[0, 'dicom_id'] = 'non_existent_dicom_id'

        # --- Catch RuntimeError, as this is what MONAI raises ---
        with pytest.raises(RuntimeError):
            dataset[0]