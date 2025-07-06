# src/ridgelet/frt.py
import numpy as np
import pywt
from skimage.transform import radon, iradon
from typing import List

class FRT:
    """
    Implements the Finite Ridgelet Transform (FRT) using the Radon transform
    and a 1D Wavelet transform.
    """

    def __init__(self, wavelet: str = 'db4'):
        """
        Initializes the FRT object.

        Args:
            wavelet (str): The name of the wavelet to use for the 1D transform,
                         as defined in the PyWavelets library.
        """
        self.wavelet = wavelet

    def forward(self, img: np.ndarray, threshold_ratio: float = 0.0) -> List[np.ndarray]:
        """
        Applies the forward Finite Ridgelet Transform.
        """
        if img.ndim != 2 or img.shape[0] != img.shape[1]:
            raise ValueError("Input image must be a 2D square array.")
        if not 0.0 <= threshold_ratio <= 1.0:
            raise ValueError("threshold_ratio must be between 0.0 and 1.0.")

        theta = np.linspace(0., 180., max(img.shape), endpoint=False)
        sinogram = radon(img, theta=theta, circle=False)

        coeffs = []
        for projection in sinogram.T:
            proj_coeffs = pywt.wavedec(projection, self.wavelet, level=None)
            coeffs.append(proj_coeffs)

        if threshold_ratio > 0:
            max_val = 0
            for proj_coeffs in coeffs:
                for detail_level in proj_coeffs[1:]:
                    max_val = max(max_val, np.max(np.abs(detail_level)))
            
            threshold = threshold_ratio * max_val

            # **Corrected line:** Use soft thresholding for cleaner results.
            for proj_coeffs in coeffs:
                for i in range(1, len(proj_coeffs)):
                    proj_coeffs[i] = pywt.threshold(proj_coeffs[i], threshold, mode='soft')
        
        return coeffs

    def inverse(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Applies the inverse Finite Ridgelet Transform.
        """
        reconstructed_sinogram = []
        for c in coeffs:
            reconstructed_sinogram.append(pywt.waverec(c, self.wavelet))
        reconstructed_sinogram = np.array(reconstructed_sinogram).T

        theta = np.linspace(0., 180., reconstructed_sinogram.shape[1], endpoint=False)
        
        reconstructed_image = iradon(reconstructed_sinogram, theta=theta, circle=False, filter_name='ramp')

        return reconstructed_image