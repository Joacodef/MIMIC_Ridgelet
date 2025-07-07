# src/data/transforms.py
import numpy as np
import torch
import pywt

from skimage.transform import resize

from monai.transforms import MapTransform
from src.ridgelet.frt import FRT

from typing import Dict, Any, Hashable, Mapping

class RidgeletTransformd(MapTransform):
    """
    A MONAI-compatible dictionary-based transform to apply the Finite Ridgelet Transform (FRT)
    to an image. This version uses a local FRT implementation.
    """

    def __init__(self, keys, threshold_ratio: float = 0.0, allow_missing_keys: bool = False):
        """
        Initializes the transform.

        Args:
            keys (list): A list of keys corresponding to the image data in the input dictionary.
            threshold_ratio (float): The ratio (0.0 to 1.0) for thresholding
                the wavelet coefficients. Defaults to 0.0 (no thresholding).
            allow_missing_keys (bool): If True, the transform will not raise an error
                                       if a key is missing from the input dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.frt = FRT(wavelet='db4')
        if not 0.0 <= threshold_ratio <= 1.0:
            raise ValueError("threshold_ratio must be between 0.0 and 1.0.")
        self.threshold_ratio = threshold_ratio

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
                
                # Use squeeze() to remove all singleton dimensions
                # (batch and channel) to get a 2D array.
                img_np = img_tensor.cpu().numpy().squeeze()
                original_size = img_np.shape[0]

                if img_np.ndim != 2 or img_np.shape[0] != img_np.shape[1]:
                    raise ValueError(f"Image must be a 2D square array for the FRT. Got shape {img_np.shape}.")

                # 1. Apply the forward and inverse Ridgelet transforms, passing the threshold ratio.
                coeffs = self.frt.forward(img_np, self.threshold_ratio)
                reconstructed_img = self.frt.inverse(coeffs)

                # 2. Resizing Logic: Center-crop or pad the reconstructed
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
    

class HaarTransformd(MapTransform):
    """
    Applies a 2D Haar wavelet transform for denoising via coefficient
    thresholding, followed by an inverse transform to reconstruct the image.
    This process is analogous to the provided RidgeletTransformd.
    """
    def __init__(self, keys: Hashable | list[Hashable] = "image", threshold_ratio: float = 0.1, allow_missing_keys: bool = False):
        """
        Initializes the transform.

        Args:
            keys (Hashable | list[Hashable]): The key(s) of the image to transform.
            threshold_ratio (float): The ratio for thresholding the detail
                coefficients. A value of 0.0 means no thresholding.
            allow_missing_keys (bool): If True, do not raise an error for missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= threshold_ratio <= 1.0:
            raise ValueError("threshold_ratio must be between 0.0 and 1.0.")
        self.threshold_ratio = threshold_ratio

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """
        Applies the Haar wavelet denoising process to the data dictionary.

        Args:
            data (Mapping[Hashable, Any]): The input data dictionary.

        Returns:
            Dict[Hashable, Any]: The transformed data dictionary.
        """
        d = dict(data)
        for key in self.keys:
            if key in d:
                img_tensor = d[key]
                img_np = img_tensor.cpu().numpy().squeeze()
                original_shape = img_np.shape

                # 1. Apply the forward 2D Haar wavelet transform
                coeffs = pywt.dwt2(img_np, 'haar')
                ll, (lh, hl, hh) = coeffs

                # 2. Threshold the detail coefficients (lh, hl, hh)
                if self.threshold_ratio > 0:
                    # Calculate a threshold value. A simple approach is to use the
                    # ratio on the maximum absolute value of all detail coeffs.
                    max_val = max(np.max(np.abs(lh)), np.max(np.abs(hl)), np.max(np.abs(hh)))
                    threshold = self.threshold_ratio * max_val
                    
                    # Apply soft thresholding
                    lh = pywt.threshold(lh, threshold, mode='soft')
                    hl = pywt.threshold(hl, threshold, mode='soft')
                    hh = pywt.threshold(hh, threshold, mode='soft')

                # 3. Apply the inverse transform to reconstruct the image
                reconstructed_img = pywt.idwt2((ll, (lh, hl, hh)), 'haar')
                
                # 4. Resize to original dimensions to handle any minor size changes
                # from the wavelet transform.
                reconstructed_img = resize(reconstructed_img, original_shape, anti_aliasing=True)

                # 5. Convert back to a tensor and add channel dimension
                transformed_tensor = torch.from_numpy(reconstructed_img).float().unsqueeze(0)
                d[key] = transformed_tensor
                
        return d