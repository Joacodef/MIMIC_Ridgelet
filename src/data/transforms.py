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
    


# Helper function for normalization
def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalizes a single channel to the [0, 1] range."""
    min_val, max_val = np.min(channel), np.max(channel)
    if max_val > min_val:
        return (channel - min_val) / (max_val - min_val)
    return np.zeros_like(channel)

class HaarTransformd(MapTransform):
    """
    Applies a multi-level Haar wavelet decomposition and stacks the coefficient
    maps into separate channels for model input.
    """
    def __init__(
        self,
        keys: List[str],
        output_key: str,
        levels: Optional[int] = None,
        threshold_ratio: float = 0.0,
        input_original_image: bool = False,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys (List[str]): Keys of the data dictionary to apply the transform to.
            output_key (str): Key for the output in the data dictionary.
            threshold_ratio (float): Ratio for soft thresholding detail coefficients. If 0, no thresholding is applied.
            input_original_image (bool): If True, concatenates the original image as the first channel.
            allow_missing_keys (bool): If True, does not raise an exception if a key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.levels = levels
        self.threshold_ratio = threshold_ratio        
        self.input_original_image = input_original_image

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue

            image_tensor = d[key]
            image_np = image_tensor.cpu().numpy().squeeze()
            original_shape = image_np.shape
            max_possible_levels = pywt.dwt_max_level(min(original_shape), 'haar')

            # 1. Determine the number of decomposition levels
            if self.levels is None or self.levels <= 0:
                num_levels = max_possible_levels
            else:
                # Use specified level, but cap it at the maximum possible
                num_levels = min(self.levels, max_possible_levels)

            if num_levels == 0:
                d[self.output_key] = image_tensor
                continue
                
            # 2. Decompose the image
            coeffs = pywt.wavedec2(image_np, 'haar', level=num_levels)
            # 3. Optional Thresholding
            if self.threshold_ratio > 0.0:
                detail_coeffs_flat = np.concatenate([np.ravel(detail) for level in coeffs[1:] for detail in level])
                if detail_coeffs_flat.size > 0:
                    threshold_value = np.percentile(np.abs(detail_coeffs_flat), self.threshold_ratio * 100)
                    thresholded_coeffs = [coeffs[0]]
                    for level_details in coeffs[1:]:
                        thresholded_coeffs.append(tuple(pywt.threshold(detail, threshold_value, mode='soft') for detail in level_details))
                    coeffs = thresholded_coeffs

            # 4. Resize all coefficient maps and collect them
            output_channels = []
            
            approx_resized = resize(coeffs[0], original_shape, anti_aliasing=True)
            output_channels.append(_normalize_channel(approx_resized))

            for level_details in coeffs[1:]:
                for detail_coeff in level_details:
                    detail_resized = resize(detail_coeff, original_shape, anti_aliasing=True)
                    output_channels.append(_normalize_channel(detail_resized))
            
            # 5. Stack channels into a single tensor
            wavelet_tensor = torch.from_numpy(np.stack(output_channels, axis=0)).float()

            # 6. Optionally concatenate the original image
            if self.input_original_image:
                final_tensor = torch.cat([image_tensor, wavelet_tensor], dim=0)
            else:
                final_tensor = wavelet_tensor
            
            d[self.output_key] = final_tensor

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