from typing import List, Union
import numpy as np
import pywt
from skimage.transform import radon, iradon

class FRT:
    """
    Implements a Finite Ridgelet Transform (FRT) using scikit-image for the
    Radon transform and PyWavelets for the 1D Wavelet Transform. This is a
    CPU-based implementation.
    """

    def __init__(self, wavelet: str = 'db4'):
        """
        Initializes the FRT processor.

        Args:
            wavelet (str): The name of the wavelet to use (e.g., 'db4').
        """
        self.wavelet = wavelet

    def forward(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Applies the forward Finite Ridgelet Transform with BayesShrink thresholding.

        Args:
            img: The input image as a NumPy array of shape (H, W).

        Returns:
            A list of NumPy arrays representing the wavelet coefficients for each projection.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        # 1. Apply Radon Transform
        theta = np.linspace(0., 180., max(img.shape), endpoint=False)
        sinogram = radon(img, theta=theta, circle=True)

        # Transpose sinogram to have projections as rows (angles, detectors)
        projections = sinogram.T

        # 2. Apply 1D Wavelet Transform to each projection
        coeffs_list = [pywt.wavedec(proj, self.wavelet, level=None) for proj in projections]

        # 3. Apply BayesShrink thresholding to each projection's coefficients
        for i in range(len(coeffs_list)):
            coeffs = coeffs_list[i]
            
            if len(coeffs) < 2:
                continue

            finest_detail_coeffs = coeffs[-1]
            median_abs_dev = np.median(np.abs(finest_detail_coeffs))
            sigma_n_sq = (median_abs_dev / 0.6745) ** 2
            
            for j in range(1, len(coeffs)):
                detail_level = coeffs[j]
                sigma_x_sq = np.var(detail_level)
                sigma_s_sq = max(0, sigma_x_sq - sigma_n_sq)
                
                if sigma_s_sq == 0:
                    threshold = np.max(np.abs(detail_level)) if detail_level.size > 0 else 0
                else:
                    threshold = sigma_n_sq / np.sqrt(sigma_s_sq)
                
                coeffs[j] = pywt.threshold(detail_level, threshold, mode='hard')

        return coeffs_list

    def inverse(self, coeffs: List[np.ndarray], original_shape: tuple) -> np.ndarray:
        """
        Applies the full inverse FRT (wavelet reconstruction + Radon reconstruction).

        Args:
            coeffs: The list of wavelet coefficients from the forward pass.
            original_shape: The (height, width) of the original image.

        Returns:
            The reconstructed image as a NumPy array.
        """
        # 1. Reconstruct the sinogram from wavelet coefficients
        sinogram = self.reconstruct_sinogram(coeffs)

        # 2. Reconstruct the image from the sinogram
        # scikit-image's iradon expects (detectors, angles), so we transpose
        reconstructed_sinogram_T = sinogram.T
        theta = np.linspace(0., 180., reconstructed_sinogram_T.shape[1], endpoint=False)
        reconstructed_img = iradon(
            reconstructed_sinogram_T,
            theta=theta,
            output_size=original_shape[0],
            circle=True,
            filter_name='shepp-logan'
        )
        
        # In case iradon returns a slightly different size, resize to original
        # This can happen if the original image was not square
        if reconstructed_img.shape != original_shape:
            from skimage.transform import resize
            reconstructed_img = resize(reconstructed_img, original_shape, anti_aliasing=True)

        return reconstructed_img

    def reconstruct_sinogram(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Performs the inverse wavelet step to reconstruct the sinogram from coefficients.

        Args:
            coeffs: The list of wavelet coefficients.

        Returns:
            The reconstructed sinogram as a NumPy array.
        """
        recon_projections = [pywt.waverec(c, self.wavelet) for c in coeffs]
        return np.array(recon_projections)