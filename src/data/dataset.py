import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class CXRFractureDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MIMIC-CXR JPG files for fracture classification.

    This class reads a CSV file containing image paths and labels, and provides
    a mechanism to load the corresponding images and apply transformations.
    """

    def __init__(self, csv_path, image_root_dir, transform=None):
        """
        Initializes the dataset.

        Args:
            csv_path (str): The full path to the CSV file (e.g., train.csv).
            image_root_dir (str): The root directory where the MIMIC-CXR 'files'
                                  folder is located.
            transform (callable, optional): A MONAI transform pipeline to be
                                            applied to each sample.
        """
        super().__init__()
        try:
            self.data_frame = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified CSV file was not found: {csv_path}")

        self.image_root_dir = image_root_dir
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            A dictionary containing the transformed image and its label,
            if dictionary-based transforms are used. Otherwise, a tuple
            of (image, label).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image information from the dataframe
        row = self.data_frame.iloc[idx]
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = row['dicom_id']
        label = torch.tensor(row['fracture'], dtype=torch.float32)

        # Construct the image file path according to the MIMIC-CXR-JPG structure
        # e.g., .../files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        p_group = "p" + subject_id[:2]
        p_id_folder = "p" + subject_id
        s_id_folder = "s" + study_id
        
        image_path = os.path.join(
            self.image_root_dir,
            p_group,
            p_id_folder,
            s_id_folder,
            f"{dicom_id}.jpg"
        )
        
        # Load the image using Pillow
        try:
            # Images are grayscale, so we load in 'L' mode
            image = Image.open(image_path).convert("L")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image_path = os.path.join(
            self.image_root_dir, p_group, p_id_folder, s_id_folder, f"{dicom_id}.jpg"
        )
        
        # Pass the path
        sample = {"image": image_path, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample