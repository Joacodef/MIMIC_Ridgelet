# src/data/transforms.py

import numpy as np
import torch
import pywt

from skimage.transform import resize

from monai.transforms import MapTransform
from src.ridgelet.frt import FRT

from typing import Dict, Any, Hashable, Mapping, List, Optional # Ensure List is imported
from scipy.ndimage import gaussian_filter

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
    Applies a multi-level Haar wavelet transform. It reconstructs the image
    from selected sub-bands and can optionally apply an Unsharp Mask filter to
    the final reconstructed image for robust sharpening.
    """
    def __init__(
        self,
        keys: List[str],
        output_key: str,
        threshold_ratio: float = 0.0,
        levels: int = 1,
        details_to_keep: Optional[List[str]] = None,
        unsharp_amount: Optional[float] = None,
        unsharp_sigma: float = 1.0,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: Keys of the data dictionary to apply the transform to.
            output_key: Key for the output in the data dictionary.
            threshold_ratio: Ratio for soft thresholding detail coefficients.
            levels: Number of wavelet decomposition levels.
            details_to_keep: List of sub-bands to keep for reconstruction.
                Options: "LL", "HL", "LH", "HH". Defaults to all.
            unsharp_amount: Scaling factor for the unsharp mask. If None or 0,
                no sharpening is applied. Typical values are between 0.5 and 1.5.
            unsharp_sigma: Standard deviation for the Gaussian blur used in
                the unsharp mask. Controls the radius of the blurring effect.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.threshold_ratio = threshold_ratio
        self.levels = levels
        self.unsharp_amount = unsharp_amount
        self.unsharp_sigma = unsharp_sigma
        # Default to keeping ALL bands if none are specified
        self.details_to_keep = details_to_keep if details_to_keep is not None else ["LL", "HL", "LH", "HH"]
        # Detail map now includes the approximation band "LL"
        self.detail_map = {"HL": 0, "LH": 1, "HH": 2}

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue

            image_tensor = d[key]
            image_np = image_tensor.cpu().numpy().squeeze()
            original_shape = image_np.shape

            # 1. Decompose the original image
            coeffs = pywt.wavedec2(image_np, 'haar', level=self.levels)

            # 2. Thresholding
            if self.threshold_ratio > 0.0:
                detail_coeffs_flat = np.concatenate([np.ravel(detail) for level in coeffs[1:] for detail in level])
                if detail_coeffs_flat.size > 0:
                    threshold_value = np.percentile(np.abs(detail_coeffs_flat), self.threshold_ratio * 100)
                else:
                    threshold_value = 0
                
                thresholded_coeffs = [coeffs[0]]
                for level_details in coeffs[1:]:
                    thresholded_coeffs.append(tuple(pywt.threshold(detail, threshold_value, mode='soft') for detail in level_details))
                coeffs = thresholded_coeffs

            # 3. Create a template for reconstruction
            recon_coeffs = [np.zeros_like(coeffs[0])] + [
                tuple(np.zeros_like(detail) for detail in level_details)
                for level_details in coeffs[1:]
            ]

            # 4. Selectively copy desired coefficients
            if "LL" in self.details_to_keep:
                recon_coeffs[0][:] = coeffs[0]

            for level_idx, level_details in enumerate(coeffs[1:]):
                for detail_name, detail_coeff_idx in self.detail_map.items():
                    if detail_name in self.details_to_keep:
                        recon_coeffs[level_idx + 1][detail_coeff_idx][:] = level_details[detail_coeff_idx]
            
            # 5. Reconstruct and resize
            reconstructed_img = pywt.waverec2(recon_coeffs, 'haar')
            reconstructed_img = resize(reconstructed_img, original_shape, anti_aliasing=True)

            # 6. Normalize the reconstructed image to [0, 1]
            min_val, max_val = np.min(reconstructed_img), np.max(reconstructed_img)
            if max_val > min_val:
                processed_img = (reconstructed_img - min_val) / (max_val - min_val)
            else:
                processed_img = np.zeros_like(reconstructed_img)

            # 7. Optional Unsharp Masking (applied AFTER normalization)
            if self.unsharp_amount is not None and self.unsharp_amount > 0:
                # Create a blurred version of the image
                blurred_img = gaussian_filter(processed_img, sigma=self.unsharp_sigma)
                # Create the mask by subtracting the blurred version from the original
                mask = processed_img - blurred_img
                # Add the scaled mask back to the original image
                sharpened_image = processed_img + self.unsharp_amount * mask
                # Clip the result to the valid [0, 1] range
                processed_img = np.clip(sharpened_image, 0, 1)

            d[self.output_key] = torch.from_numpy(processed_img).float().unsqueeze(0)

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