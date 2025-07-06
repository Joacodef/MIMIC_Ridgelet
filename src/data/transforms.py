# src/data/transforms.py
import numpy as np
import torch
from monai.transforms import MapTransform
from src.ridgelet.frt import FRT

class RidgeletTransformd(MapTransform):
    """
    A MONAI-compatible dictionary-based transform to apply the Finite Ridgelet Transform (FRT)
    to an image. This version uses a local FRT implementation.
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        """
        Initializes the transform.

        Args:
            keys (list): A list of keys corresponding to the image data in the input dictionary.
            allow_missing_keys (bool): If True, the transform will not raise an error
                                       if a key is missing from the input dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.frt = FRT(wavelet='db4')

    def __call__(self, data):
        """
        Applies the Ridgelet transform and inverse transform to the image data
        for the specified keys.

        Args:
            data (dict): The input dictionary from the MONAI dataset, containing image tensors.

        Returns:
            dict: The dictionary with the Ridgelet-processed image.
        """
        d = dict(data)
        for key in self.keys:
            if key in d:
                img_tensor = d[key]
                if not isinstance(img_tensor, torch.Tensor):
                    raise TypeError("Input must be a PyTorch tensor.")
                
                # **Corrected line**: Use squeeze() to remove all singleton dimensions
                # (batch and channel) to get a 2D array.
                img_np = img_tensor.cpu().numpy().squeeze()
                original_size = img_np.shape[0]

                if img_np.ndim != 2 or img_np.shape[0] != img_np.shape[1]:
                    raise ValueError(f"Image must be a 2D square array for the FRT. Got shape {img_np.shape}.")

                # 1. Apply the forward and inverse Ridgelet transforms
                coeffs = self.frt.forward(img_np)
                reconstructed_img = self.frt.inverse(coeffs)

                # 2. Corrected Resizing Logic: Center-crop or pad the reconstructed
                #    image to match the original input size.
                current_h, current_w = reconstructed_img.shape
                
                h_diff = original_size - current_h
                pad_top = h_diff // 2
                pad_bottom = h_diff - pad_top
                
                w_diff = original_size - current_w
                pad_left = w_diff // 2
                pad_right = w_diff - pad_left

                if h_diff > 0 or w_diff > 0:
                    # Pad the image if it's smaller
                    padding = ((pad_top, pad_bottom), (pad_left, pad_right))
                    reconstructed_img = np.pad(reconstructed_img, padding, mode='constant', constant_values=0)
                
                if h_diff < 0 or w_diff < 0:
                    # Crop the image if it's larger
                    crop_top = -pad_top
                    crop_bottom = current_h - (-pad_bottom)
                    crop_left = -pad_left
                    crop_right = current_w - (-pad_right)
                    reconstructed_img = reconstructed_img[crop_top:crop_bottom, crop_left:crop_right]

                if reconstructed_img.shape != (original_size, original_size):
                     from skimage.transform import resize
                     reconstructed_img = resize(reconstructed_img, (original_size, original_size), anti_aliasing=True)

                # Convert back to a tensor and restore the channel dimension
                transformed_tensor = torch.from_numpy(reconstructed_img).float().unsqueeze(0)
                d[key] = transformed_tensor
        return d