# src/data/transforms.py

import numpy as np
import torch
import pywt

from skimage.transform import resize

from monai.transforms import MapTransform
from src.ridgelet.frt import FRT

from typing import Dict, Any, Hashable, Mapping, List # Ensure List is imported

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

                # The `resize` import at the top level is sufficient, no need for nested import
                if reconstructed_img.shape != (original_size, original_size):
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
    Returns both the original and the reconstructed Haar image.
    """
    def __init__(self, keys: Hashable | list[Hashable] = "image", threshold_ratio: float = 0.1, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= threshold_ratio <= 1.0:
            raise ValueError("threshold_ratio must be between 0.0 and 1.0.")
        self.threshold_ratio = threshold_ratio

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                img_tensor = d[key]
                
                # Store the original image tensor for multi-channel output
                original_img_tensor = img_tensor.clone()

                img_np = img_tensor.cpu().numpy().squeeze()
                original_shape = img_np.shape

                # 1. Apply the forward 2D Haar wavelet transform
                coeffs = pywt.dwt2(img_np, 'haar')
                ll, (lh, hl, hh) = coeffs

                # 2. Threshold the detail coefficients (lh, hl, hh)
                if self.threshold_ratio > 0:
                    # Calculate a threshold value based on the maximum absolute value
                    # of all detail coefficients to ensure a consistent threshold across bands.
                    max_val_lh = np.max(np.abs(lh)) if lh.size > 0 else 0
                    max_val_hl = np.max(np.abs(hl)) if hl.size > 0 else 0
                    max_val_hh = np.max(np.abs(hh)) if hh.size > 0 else 0
                    
                    max_overall_detail_val = max(max_val_lh, max_val_hl, max_val_hh)
                    
                    if max_overall_detail_val > 0:
                        threshold = self.threshold_ratio * max_overall_detail_val
                    else:
                        threshold = 0 # No details to threshold if all are zero

                    # Apply soft thresholding
                    lh = pywt.threshold(lh, threshold, mode='soft')
                    hl = pywt.threshold(hl, threshold, mode='soft')
                    hh = pywt.threshold(hh, threshold, mode='soft')

                # 3. Apply the inverse transform to reconstruct the image
                reconstructed_img = pywt.idwt2((ll, (lh, hl, hh)), 'haar')
                
                # 4. Resize to original dimensions to handle any minor size changes
                # from the wavelet transform.
                # Ensure the reconstructed image is within float type range before resizing
                reconstructed_img = reconstructed_img.astype(np.float32)
                reconstructed_img = resize(reconstructed_img, original_shape, anti_aliasing=True)

                # 5. Convert reconstructed image back to a tensor and add channel dimension
                transformed_tensor = torch.from_numpy(reconstructed_img).float().unsqueeze(0)
                
                # Update the dictionary with both original and transformed image
                d[key] = original_img_tensor  # Keep original image under its key
                d[f"{key}_haar"] = transformed_tensor # Add Haar transformed image under a new key
                
        return d

class ConcatenateChannelsd(MapTransform):
    """
    Concatenates multiple image tensors along the channel dimension.
    Assumes all input images have a channel dimension (C, H, W).
    """
    def __init__(self, keys: List[Hashable], output_key: Hashable, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        if len(keys) < 2:
            raise ValueError("ConcatenateChannelsd requires at least two keys to concatenate.")
        self.output_key = output_key

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        tensors_to_concat = []
        for key in self.keys:
            if key not in d and not self.allow_missing_keys:
                raise KeyError(f"Missing key '{key}' in data, and allow_missing_keys is False.")
            if key in d:
                tensors_to_concat.append(d[key])
            
        if not tensors_to_concat:
            return d # No tensors to concatenate, return original data

        # Concatenate along the channel dimension (dim=0 assuming C, H, W)
        d[self.output_key] = torch.cat(tensors_to_concat, dim=0)
        
        # Optionally remove individual keys if they are no longer needed
        for key in self.keys:
            if key != self.output_key: # Don't delete the new key if it's one of the original keys
                d.pop(key, None)
        
        return d