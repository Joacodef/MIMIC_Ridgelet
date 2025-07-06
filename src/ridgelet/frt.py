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

    def forward(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Applies the forward Finite Ridgelet Transform.

        Args:
            img (np.ndarray): A 2D NumPy array representing the image.
                               The image must be square.

        Returns:
            list: A list where each element is a list of Ridgelet coefficients
                  for a projection.
        """
        if img.ndim != 2 or img.shape[0] != img.shape[1]:
            raise ValueError("Input image must be a 2D square array.")

        theta = np.linspace(0., 180., max(img.shape), endpoint=False)
        sinogram = radon(img, theta=theta, circle=False)

        coeffs = []
        for projection in sinogram.T:
            coeffs.append(pywt.wavedec(projection, self.wavelet, level=None))

        return coeffs

    def inverse(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Applies the inverse Finite Ridgelet Transform.

        Args:
            coeffs (list): The list of Ridgelet coefficients obtained from the
                           forward transform.

        Returns:
            np.ndarray: The reconstructed 2D image.
        """
        reconstructed_sinogram = []
        for c in coeffs:
            reconstructed_sinogram.append(pywt.waverec(c, self.wavelet))
        reconstructed_sinogram = np.array(reconstructed_sinogram).T

        # **Corrected line:** The number of angles for theta must match the
        # number of columns in the sinogram (shape[1]).
        theta = np.linspace(0., 180., reconstructed_sinogram.shape[1], endpoint=False)
        
        reconstructed_image = iradon(reconstructed_sinogram, theta=theta, circle=False, filter_name='ramp')

        return reconstructed_image