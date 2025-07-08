# src/ridgelet/w_packet_frt.py

import numpy as np
import pywt
from skimage.transform import radon, iradon
from typing import List, Tuple

# Import the À Trous decomposition function from your transforms file
from src.data.transforms import a_trous_decomposition

def neighshrink_threshold(coeffs: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Applies NeighShrink thresholding to wavelet coefficients.

    This is a placeholder for the full implementation of NeighShrink with SURE.
    """
    # Placeholder: For now, we can implement a simple universal threshold
    # The full implementation would be significantly more complex.
    sigma_n_sq = (np.median(np.abs(coeffs)) / 0.6745) ** 2
    threshold = np.sqrt(sigma_n_sq) * np.sqrt(2 * np.log(len(coeffs)))
    return pywt.threshold(coeffs, threshold, mode='soft')


class WPacketFRT:
    """
    Implements a Finite Ridgelet Transform using Wavelet Packets (WPT) and
    the À Trous algorithm, following the methodology from the reference paper.
    """

    def __init__(self, wavelet: str = 'db4', a_trous_scales: int = 3, wpt_level: int = 3):
        """
        Initializes the WPacketFRT processor.

        Args:
            wavelet (str): The name of the wavelet to use for the WPT.
            a_trous_scales (int): The number of scales for À Trous decomposition.
            wpt_level (int): The decomposition level for the Wavelet Packet Transform.
        """
        self.wavelet = wavelet
        self.a_trous_scales = a_trous_scales
        self.wpt_level = wpt_level
        self.theta = None

    def forward(self, img: np.ndarray) -> Tuple[List[list], np.ndarray]:
        """
        Applies the forward transform using À Trous, Radon, and WPT.

        Steps:
        1. Decompose the image using the À Trous algorithm.
        2. For each detail layer:
           a. Apply the Radon Transform.
           b. Apply the Wavelet Packet Transform to the projections.
           c. Apply NeighShrink thresholding to the WPT coefficients.
        3. Return the processed coefficients and the final residual layer.
        """
        if img.ndim != 2:
            raise ValueError("Input image must be a 2D NumPy array.")
            
        if self.theta is None:
             self.theta = np.linspace(0., 180., max(img.shape), endpoint=False)

        # 1. Decompose the image using the À Trous algorithm
        layers = a_trous_decomposition(img, scales=self.a_trous_scales)
        detail_layers = layers[:-1]
        residual_layer = layers[-1]

        all_processed_coeffs = []
        # 2. Loop through each detail layer
        for detail_layer in detail_layers:
            # 2a. Apply Radon Transform
            sinogram = radon(detail_layer, theta=self.theta, circle=True)
            projections = sinogram.T
            
            layer_coeffs = []
            for proj in projections:
                # 2b. Apply Wavelet Packet Transform
                wp = pywt.WaveletPacket(data=proj, wavelet=self.wavelet, mode='symmetric', maxlevel=self.wpt_level)
                nodes = [node.path for node in wp.get_level(self.wpt_level, order='natural')]
                
                thresholded_nodes = []
                for node_path in nodes:
                    # 2c. Apply NeighShrink thresholding
                    original_coeffs = wp[node_path].data
                    thr_coeffs = neighshrink_threshold(original_coeffs)
                    thresholded_nodes.append((node_path, thr_coeffs))
                
                layer_coeffs.append(thresholded_nodes)
            
            all_processed_coeffs.append(layer_coeffs)

        # 3. Return the collection of processed coefficients and the residual
        return all_processed_coeffs, residual_layer

    def reconstruct(self, processed_coeffs: list, residual: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Reconstructs the image from the processed coefficients and residual.

        Steps:
        1. For each processed detail layer's coefficients:
           a. Apply the inverse Wavelet Packet Transform to reconstruct projections.
           b. Apply the inverse Radon Transform to the reconstructed sinogram.
        2. Sum the reconstructed detail layers.
        3. Add the residual layer back to the sum.
        """
        reconstructed_details = []

        # Ensure theta is available, which should be set during the forward pass.
        # If not, recalculate it based on the original image shape.
        if not hasattr(self, 'theta') or self.theta is None:
            self.theta = np.linspace(0., 180., max(original_shape), endpoint=False)

        # 1. Loop through each processed detail layer's coefficients
        for layer_coeffs in processed_coeffs:
            reconstructed_projections = []
            
            # Reconstruct each projection from its WPT coefficients
            for proj_coeffs in layer_coeffs:
                # Create a new WaveletPacket object to hold the coefficients
                wp_recon = pywt.WaveletPacket(data=None, wavelet=self.wavelet, mode='symmetric', maxlevel=self.wpt_level)
                
                for node_path, thr_coeffs in proj_coeffs:
                    try:
                        wp_recon[node_path] = thr_coeffs
                    except ValueError:
                        # This can happen if the coefficient length is inconsistent.
                        # For robustness, we can skip problematic nodes.
                        print(f"Warning: Skipping node {node_path} due to size mismatch.")
                        continue

                # Reconstruct the projection from the coefficients
                reconstructed_proj = wp_recon.reconstruct(update=False)
                reconstructed_projections.append(reconstructed_proj)

            # Form the sinogram from the list of reconstructed projections
            # The .T operation is reversed from the forward pass
            reconstructed_sinogram = np.array(reconstructed_projections).T
            
            # 1b. Apply the inverse Radon Transform
            # The filter is disabled as filtering is handled by the wavelet thresholding
            reconstructed_layer = iradon(
                reconstructed_sinogram,
                theta=self.theta,
                output_size=max(original_shape), # Assumes square output, common for medical imaging
                circle=True,
                filter_name=None
            )
            reconstructed_details.append(reconstructed_layer)

        # 2. Sum the reconstructed detail layers
        # Ensure all layers have the same shape before summing
        min_size = min(layer.shape[0] for layer in reconstructed_details)
        reconstructed_details_resized = [layer[:min_size, :min_size] for layer in reconstructed_details]
        reconstructed_sum = np.sum(reconstructed_details_resized, axis=0)

        # 3. Add the residual layer back
        # Ensure residual has the same shape as the sum of details
        if reconstructed_sum.shape != residual.shape:
             from skimage.transform import resize
             residual_resized = resize(residual, reconstructed_sum.shape, mode='reflect', anti_aliasing=True)
             final_image = reconstructed_sum + residual_resized
        else:
             final_image = reconstructed_sum + residual

        # Ensure the final output matches the original image shape
        if final_image.shape != original_shape:
            from skimage.transform import resize
            final_image = resize(final_image, original_shape, mode='reflect', anti_aliasing=True)
            
        return final_image