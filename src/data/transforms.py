# src/data/transforms.py

import numpy as np
import torch
import ptwt, pywt
import math

from skimage.transform import resize
import torch.nn.functional as F

from scipy.signal import convolve2d

from monai.transforms import MapTransform
from src.ridgelet.frt import FRT

from typing import Dict, Any, Hashable, Mapping, List, Optional
from scipy.ndimage import gaussian_filter


# Helper function for normalization
def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalizes a single channel to the [0, 1] range."""
    min_val, max_val = np.min(channel), np.max(channel)
    if max_val > min_val:
        return (channel - min_val) / (max_val - min_val)
    return np.zeros_like(channel)

def _normalize_channel_torch(channel: torch.Tensor) -> torch.Tensor:
    """Normalizes a single channel tensor to the [0, 1] range."""
    min_val, max_val = torch.min(channel), torch.max(channel)
    if max_val > min_val:
        return (channel - min_val) / (max_val - min_val)
    return torch.zeros_like(channel)

def dwt_max_level(data_len: int, wavelet: str) -> int:
    """
    Calculates the maximum level of wavelet decomposition using pywt.
    The main transform is still done with ptwt on the GPU; this helper
    is a lightweight way to get wavelet properties.

    Args:
        data_len (int): The length of the data/image dimension.
        wavelet (str): The name of the wavelet.

    Returns:
        int: The maximum number of decomposition levels.
    """
    try:
        # Use the pywt library to create a wavelet object
        w = pywt.Wavelet(wavelet)
        filter_len = w.dec_len
    except ValueError:
        # Fallback for unrecognized wavelets
        filter_len = 2  # Haar's filter length as a safe default

    if filter_len <= 1:
        return 0
    
    # Formula is equivalent to the one used in PyWavelets
    return int(math.log2(data_len / (filter_len - 1)))

def soft_threshold_torch(data: torch.Tensor, value: float) -> torch.Tensor:
    """
    Performs soft thresholding on a PyTorch tensor, replicating
    pywt.threshold(data, value, mode='soft').

    Args:
        data (torch.Tensor): The input tensor.
        value (float): The threshold value.

    Returns:
        torch.Tensor: The thresholded tensor.
    """
    return torch.sign(data) * F.relu(torch.abs(data) - value)

class WaveletTransformd(MapTransform):
    """
    Applies a multi-level wavelet decomposition on the GPU and stacks the coefficient
    maps into separate channels for model input. This transform is batch-aware.
    """
    def __init__(
        self,
        keys: List[str],
        output_key: str,
        wavelet_name: str = 'haar',
        levels: Optional[int] = None,
        threshold_ratio: float = 0.0,
        input_original_image: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.wavelet_name = wavelet_name
        self.levels = levels
        self.threshold_ratio = threshold_ratio
        self.input_original_image = input_original_image
        self.device = torch.device(device)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue

            image_batch = d[key].to(self.device)
            processed_images = []

            # Loop over each image in the batch
            for single_image_tensor in image_batch:
                img_to_process = single_image_tensor.squeeze()
                if img_to_process.ndim != 2:
                    raise ValueError(f"Could not extract a 2D image. Input slice shape: {img_to_process.shape}.")

                original_shape = img_to_process.shape
                
                max_possible_levels = dwt_max_level(min(original_shape), self.wavelet_name)
                num_levels = min(self.levels, max_possible_levels) if self.levels is not None and self.levels > 0 else max_possible_levels

                if num_levels == 0:
                    processed_images.append(single_image_tensor)
                    continue

                coeffs = ptwt.wavedec2(img_to_process, self.wavelet_name, level=num_levels)
                
                if self.threshold_ratio > 0.0:
                    detail_coeffs_flat = torch.cat([torch.ravel(detail) for c in coeffs[1:] for detail in c])
                    if detail_coeffs_flat.numel() > 0:
                        threshold_value = torch.quantile(torch.abs(detail_coeffs_flat), self.threshold_ratio)
                        thresholded_coeffs = [coeffs[0]]
                        for level_details in coeffs[1:]:
                            thresholded_coeffs.append(tuple(soft_threshold_torch(detail, threshold_value) for detail in level_details))
                        coeffs = thresholded_coeffs

                output_channels = []
                
                # Resize and normalize coefficients
                all_coeffs = [coeffs[0]] + [detail for level in coeffs[1:] for detail in level]
                for coeff in all_coeffs:
                    if coeff.ndim == 2:
                        coeff_4d = coeff.unsqueeze(0).unsqueeze(0)
                    else:
                        coeff_4d = coeff.unsqueeze(0)
                        
                    resized_coeff = F.interpolate(coeff_4d, size=original_shape, mode='bilinear', align_corners=False).squeeze()
                    output_channels.append(_normalize_channel_torch(resized_coeff))
                
                wavelet_tensor = torch.stack(output_channels, dim=0)

                if self.input_original_image:
                    # Robustly reshape original image to (1, H, W) for concatenation
                    original_img_for_cat = single_image_tensor.view(1, *original_shape)
                    final_tensor = torch.cat([original_img_for_cat, wavelet_tensor], dim=0)
                else:
                    final_tensor = final_tensor = wavelet_tensor
                
                processed_images.append(final_tensor)

            # Stack the list of processed images into a single batch tensor
            output_batch = torch.stack(processed_images)
            
            # Use the specified output_key
            d[self.output_key] = output_batch
            
            # Clean up the original key if it's different from the output key
            if key != self.output_key:
                del d[key]

        return d
    

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