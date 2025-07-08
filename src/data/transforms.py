# src/data/transforms.py

import numpy as np
import torch
import pywt

from skimage.transform import resize

from scipy.signal import convolve2d

from monai.transforms import MapTransform
from src.ridgelet.frt import FRT

from typing import Dict, Any, Hashable, Mapping, List, Optional
from scipy.ndimage import gaussian_filter


class RidgeletTransformd(MapTransform):
    """
    A MONAI-compatible dictionary-based transform to apply the Finite Ridgelet Transform (FRT).
    This transform uses a CPU-based backend (scikit-image and PyWavelets) and
    applies BayesShrink adaptive thresholding for denoising.
    """

    def __init__(self, keys: List[str], output_type: str = 'reconstruction', allow_missing_keys: bool = False):
        """
        Initializes the transform.

        Args:
            keys (list): Keys of the data dictionary to apply the transform to.
            output_type (str): The desired output from the transform. Must be either
                               'reconstruction' (for the denoised image) or 'sinogram'
                               (for the denoised Radon transform).
            allow_missing_keys (bool): If True, do not raise an error for missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        
        # This transform is CPU-only, so the device parameter is removed.
        self.frt = FRT(wavelet='sym8')

        if output_type not in ['reconstruction', 'sinogram']:
            raise ValueError("output_type must be 'reconstruction' or 'sinogram'.")
        self.output_type = output_type

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                img_tensor = d[key]
                original_shape = img_tensor.shape[-2:]
                
                # --- Convert to NumPy for CPU processing ---
                img_np = img_tensor.squeeze().cpu().numpy()

                # --- Apply the NumPy-based transform ---
                coeffs = self.frt.forward(img_np)

                if self.output_type == 'sinogram':
                    processed_output = self.frt.reconstruct_sinogram(coeffs)
                else:  # 'reconstruction'
                    processed_output = self.frt.inverse(coeffs, original_shape)

                # --- Resize and Normalize using NumPy ---
                resized_output = resize(processed_output, original_shape, anti_aliasing=True)

                min_val, max_val = np.min(resized_output), np.max(resized_output)
                if max_val > min_val:
                    normalized_output = (resized_output - min_val) / (max_val - min_val)
                else:
                    normalized_output = np.zeros_like(resized_output)
                
                # --- Convert back to PyTorch Tensor ---
                # Add a channel dimension for MONAI compatibility
                output_tensor = torch.from_numpy(normalized_output).float().unsqueeze(0)
                d[key] = output_tensor
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
    

# B3 spline kernel for the À Trous algorithm, as defined in the reference paper.
# Source: D. Gupta et al. / Optik 125 (2014) 1417-1422
B3_SPLINE_KERNEL = (1/256) * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

def a_trous_decomposition(image: np.ndarray, scales: int) -> list[np.ndarray]:
    """
    Decomposes an image using the À Trous (with holes) algorithm.

    This algorithm separates the image into multiple wavelet planes (detail layers)
    at different scales and a final residual layer.

    Args:
        image (np.ndarray): The input image as a 2D NumPy array.
        scales (int): The number of decomposition scales.

    Returns:
        list[np.ndarray]: A list of NumPy arrays containing the detail layers
                          for each scale, with the final residual layer as the
                          last element.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise TypeError("Input must be a 2D NumPy array.")
    if not isinstance(scales, int) or scales <= 0:
        raise ValueError("Scales must be a positive integer.")

    detail_layers = []
    current_image = image.copy()

    for _ in range(scales):
        # Convolve the current image with the B3 spline kernel to get the smoothed version
        smoothed_image = convolve2d(current_image, B3_SPLINE_KERNEL, mode='same', boundary='symm')
        
        # The detail layer is the difference between the current and smoothed images
        detail = current_image - smoothed_image
        detail_layers.append(detail)
        
        # The new image for the next iteration is the smoothed image
        current_image = smoothed_image

    # The last layer is the residual (the final smoothed image)
    detail_layers.append(current_image)
    
    return detail_layers