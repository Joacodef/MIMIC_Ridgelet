from typing import List

import numpy as np
import pywt
from skimage.transform import iradon as cpu_iradon
from skimage.transform import radon as cpu_radon

try:
    import astra
    import cupy
    import ptwt
    import torch
    GPU_LIBS_AVAILABLE = True
except ImportError:
    GPU_LIBS_AVAILABLE = False


class FRT:
    """
    Implements a device-aware Finite Ridgelet Transform (FRT) using the Radon
    transform and a 1D Wavelet transform. Supports both CPU (scikit-image)
    and GPU (astra-toolbox) execution.
    """

    def __init__(self, wavelet: str = 'db4', device: str = 'cpu'):
        self.wavelet = wavelet
        self.device = device

        if self.device == 'cuda' and not GPU_LIBS_AVAILABLE:
            raise ImportError(
                "GPU libraries (torch, cupy, ptwt, astra-toolbox) are not installed. "
                "Please install them or use device='cpu'."
            )

    def forward(self, img: np.ndarray or torch.Tensor, sparsity_level: float = 1.0) -> List:
        """
        Applies the forward Finite Ridgelet Transform using rank-based hard thresholding.

        Args:
            img: The input image as a NumPy array or PyTorch Tensor.
            sparsity_level (float): The fraction of detail coefficients to keep. 1.0 keeps all.
        """
        if self.device == 'cuda':
            return self._forward_gpu(img, sparsity_level)
        else:
            return self._forward_cpu(img, sparsity_level)

    def inverse(self, coeffs: List, original_shape: tuple) -> np.ndarray or torch.Tensor:
        """
        Applies the full inverse FRT (wavelet reconstruction + Radon reconstruction).
        """
        # Step 1: Reconstruct the sinogram from wavelet coefficients
        reconstructed_sinogram = self.reconstruct_sinogram(coeffs)

        # Step 2: Reconstruct the image from the sinogram
        if self.device == 'cuda':
            return self._inverse_radon_gpu(reconstructed_sinogram, original_shape)
        else:
            return self._inverse_radon_cpu(reconstructed_sinogram, original_shape)
        
    def reconstruct_sinogram(self, coeffs: List) -> np.ndarray or torch.Tensor:
        """
        Performs only the inverse wavelet step to reconstruct the sinogram from coefficients.
        This is the direct representation of the extracted features.
        """
        if self.device == 'cuda':
            # ptwt.waverec returns a tensor of shape (angles, detectors)
            return ptwt.waverec(coeffs, self.wavelet, axis=-1).contiguous()
        else:
            # pywt.waverec operates on a list of arrays. We combine them
            # to get the conventional (angles, detectors) shape.
            recon_projections = [pywt.waverec(c, self.wavelet) for c in coeffs]
            return np.array(recon_projections)

    # --- CPU Methods ---
    def _forward_cpu(self, img: np.ndarray, sparsity_level: float) -> List[np.ndarray]:
        if not isinstance(img, np.ndarray):
            raise TypeError("CPU mode requires a NumPy ndarray as input.")
        if img.ndim != 2 or img.shape[0] != img.shape[1]:
            raise ValueError("Input image must be a 2D square array.")

        theta = np.linspace(0., 180., max(img.shape), endpoint=False)
        sinogram = cpu_radon(img, theta=theta, circle=True)

        projections = sinogram.T
        coeffs_list = [pywt.wavedec(
            proj, self.wavelet, level=None) for proj in projections]

        if sparsity_level < 1.0:
            for i in range(len(coeffs_list)):
                coeffs = coeffs_list[i]
                detail_coeffs = np.concatenate([c for c in coeffs[1:]])
                if detail_coeffs.size == 0:
                    continue

                q = 100 * (1.0 - sparsity_level)
                threshold = np.percentile(np.abs(detail_coeffs), q)

                for j in range(1, len(coeffs)):
                    coeffs[j] = pywt.threshold(
                        coeffs[j], threshold, mode='hard')

        return coeffs_list

    def _inverse_radon_cpu(self, sinogram: np.ndarray, original_shape: tuple) -> np.ndarray:
        # Input sinogram shape: (angles, detectors)
        # cpu_iradon expects (detectors, angles), so we transpose
        reconstructed_sinogram_T = sinogram.T
        theta = np.linspace(0., 180., reconstructed_sinogram_T.shape[1], endpoint=False)
        return cpu_iradon(reconstructed_sinogram_T, theta=theta, circle=True, filter_name='Shepp-Logan')

    # --- GPU Methods ---
    def _forward_gpu(self, img_tensor: torch.Tensor, sparsity_level: float) -> List[torch.Tensor]:
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("GPU mode requires a PyTorch Tensor as input.")
        img_tensor = img_tensor.squeeze().to(self.device).contiguous()
        height, width = img_tensor.shape

        radius = min(height, width) // 2
        y, x = torch.meshgrid(
            torch.arange(-radius, radius, device=self.device),
            torch.arange(-radius, radius, device=self.device),
            indexing='ij'
        )
        mask = (x*x + y*y <= radius*radius).to(self.device)
        center = height // 2
        img_tensor = img_tensor[center-radius:center+radius, center-radius:center+radius] * mask
        
        vol_geom = astra.create_vol_geom(img_tensor.shape[1], img_tensor.shape[0])
        detector_width = int(np.sqrt(2) * max(img_tensor.shape)) + 1
        num_angles = img_tensor.shape[1]
        angles = np.linspace(0.0, np.pi, num_angles, endpoint=False)
        proj_geom = astra.create_proj_geom('parallel', 1.0, detector_width, angles)

        img_id = astra.data2d.create('-vol', vol_geom, data=cupy.asnumpy(img_tensor))
        sinogram_id = astra.data2d.create('-sino', proj_geom)

        fp_config = astra.astra_dict('FP_CUDA')
        fp_config['VolumeDataId'] = img_id
        fp_config['ProjectionDataId'] = sinogram_id
        
        alg_id = astra.algorithm.create(fp_config)
        astra.algorithm.run(alg_id)
        
        sinogram_cupy = astra.data2d.get(sinogram_id)
        # Shape: (num_angles, detector_pixels)
        sinogram_for_wavelet = torch.as_tensor(sinogram_cupy, device=self.device).T.contiguous()
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(img_id)
        astra.data2d.delete(sinogram_id)
        
        # Coeffs is a list of tensors, where each tensor is a wavelet level
        coeffs = ptwt.wavedec(sinogram_for_wavelet, self.wavelet, level=None, axis=-1)

        if sparsity_level < 1.0:
            # We must threshold each projection independently, not globally.
            
            # This will hold the final thresholded coefficients
            thresholded_coeffs = [coeffs[0].clone()] 
            
            # Temporarily combine all detail levels for easy iteration
            all_detail_coeffs_by_level = [c.clone() for c in coeffs[1:]]

            # Iterate through each projection angle
            for i in range(sinogram_for_wavelet.shape[0]):
                # Gather all detail coefficients for this single projection (i)
                proj_coeffs = [level[i] for level in all_detail_coeffs_by_level]
                proj_detail_coeffs_flat = torch.cat([c.flatten() for c in proj_coeffs])
                
                if proj_detail_coeffs_flat.numel() == 0:
                    continue

                # Calculate the threshold for this projection only
                q = 1.0 - sparsity_level
                threshold = torch.quantile(torch.abs(proj_detail_coeffs_flat), q)

                # Apply the threshold to the actual coefficient levels for this projection
                for j in range(len(all_detail_coeffs_by_level)):
                    level_coeffs = all_detail_coeffs_by_level[j][i]
                    all_detail_coeffs_by_level[j][i] = level_coeffs * (torch.abs(level_coeffs) > threshold)

            thresholded_coeffs.extend(all_detail_coeffs_by_level)
            return thresholded_coeffs
        
        return coeffs

    def _inverse_radon_gpu(self, sinogram: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        # Input sinogram shape: (angles, detectors)
        height, width = original_shape
        vol_geom = astra.create_vol_geom(width, height)
        
        num_angles, detector_width = sinogram.shape
        angles = np.linspace(0.0, np.pi, num_angles, endpoint=False)
        proj_geom = astra.create_proj_geom('parallel', 1.0, detector_width, angles)

        recon_id = astra.data2d.create('-vol', vol_geom)
        
        # ASTRA's create function expects data in (angles, detectors) format, which is correct
        sinogram_data = cupy.asnumpy(sinogram)
        sinogram_id = astra.data2d.create('-sino', proj_geom, data=sinogram_data)

        fbp_config = astra.astra_dict('FBP_CUDA')
        fbp_config['ReconstructionDataId'] = recon_id
        fbp_config['ProjectionDataId'] = sinogram_id
        fbp_config['FilterType'] = 'Shepp-Logan'
        alg_id = astra.algorithm.create(fbp_config)
        astra.algorithm.run(alg_id)

        reconstructed_image_cupy = astra.data2d.get(recon_id)
        reconstructed_tensor = torch.as_tensor(reconstructed_image_cupy, device=self.device)

        astra.algorithm.delete(alg_id)
        astra.algorithm.delete(recon_id)
        astra.data2d.delete(sinogram_id)

        return reconstructed_tensor