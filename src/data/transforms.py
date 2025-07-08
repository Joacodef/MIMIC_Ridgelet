# src/data/transforms.py

import numpy as np
import torch
import pywt

from skimage.transform import resize

from monai.transforms import MapTransform
from src.ridgelet.frt import FRT

from typing import Dict, Any, Hashable, Mapping, List, Optional
from scipy.ndimage import gaussian_filter


class RidgeletTransformd(MapTransform):
    """
    A MONAI-compatible dictionary-based transform to apply the Finite Ridgelet Transform (FRT).
    This transform uses a rank-based hard threshold to extract a specific percentage of
    the most significant linear features.
    """

    def __init__(self, keys: List[str], sparsity_level: float = 1.0, output_type: str = 'reconstruction', device: str = 'cpu', allow_missing_keys: bool = False):
        """
        Initializes the transform.

        Args:
            keys (list): Keys of the data dictionary to apply the transform to.
            sparsity_level (float): The fraction of the most significant wavelet detail
                                    coefficients to KEEP. Must be between 0.0 and 1.0.
                                    For example, 0.05 keeps the top 5%.
            device (str): The device to run on, either 'cpu' or 'cuda'.
            allow_missing_keys (bool): If True, do not raise an error for missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= sparsity_level <= 1.0:
            raise ValueError("sparsity_level must be between 0.0 and 1.0.")
            
        self.sparsity_level = sparsity_level
        self.device = device
        self.frt = FRT(wavelet='db4', device=device)

        if output_type not in ['reconstruction', 'sinogram']:
            raise ValueError("output_type must be 'reconstruction' or 'sinogram'.")
        self.output_type = output_type

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                img_tensor = d[key]
                original_shape = img_tensor.shape[-2:]
                
                processed_img = img_tensor.squeeze()

                if self.device == 'cuda':
                    processed_output = self._apply_gpu(processed_img, original_shape)
                else:
                    processed_output = self._apply_cpu(processed_img.cpu().numpy(), original_shape)
                
                # Add a channel dimension back for MONAI compatibility
                d[key] = processed_output.unsqueeze(0)
        return d


    def _apply_transform(self, img_data, original_shape):
        """Helper to apply the correct transform based on output_type."""
        coeffs = self.frt.forward(img_data, self.sparsity_level)

        if self.output_type == 'sinogram':
            return self.frt.reconstruct_sinogram(coeffs)
        else: # 'reconstruction'
            return self.frt.inverse(coeffs, original_shape)
        
        
    def _apply_cpu(self, img_np: np.ndarray, original_shape: tuple) -> torch.Tensor:
        """Applies the transform using CPU libraries."""
        processed_output = self._apply_transform(img_np, original_shape)

        # The sinogram and reconstruction might have different shapes, so we resize
        resized_output = resize(processed_output, original_shape, anti_aliasing=True)
        
        min_val, max_val = np.min(resized_output), np.max(resized_output)
        if max_val > min_val:
            normalized_output = (resized_output - min_val) / (max_val - min_val)
        else:
            normalized_output = np.zeros_like(resized_output)

        return torch.from_numpy(normalized_output).float()
    

    def _apply_gpu(self, img_tensor: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """Applies the transform using GPU libraries."""
        processed_output = self._apply_transform(img_tensor, original_shape)
        
        # Add dimensions for PyTorch interpolation
        resized_output = processed_output.unsqueeze(0).unsqueeze(0)
        resized_output = torch.nn.functional.interpolate(
            resized_output, size=original_shape, mode='bilinear', align_corners=False
        ).squeeze()

        min_val, max_val = torch.min(resized_output), torch.max(resized_output)
        if max_val > min_val:
            normalized_output = (resized_output - min_val) / (max_val - min_val)
        else:
            normalized_output = torch.zeros_like(resized_output)
            
        return normalized_output.float()

    


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