import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# By running pytest from the project root, it will automatically find the `src` module.
from src.data.transforms import (
    _normalize_channel,
    WaveletTransformd,
    RidgeletTransformd,
    a_trous_decomposition,
    B3_SPLINE_KERNEL
)

# --- Test Fixtures ---

@pytest.fixture
def sample_image_tensor():
    """Provides a sample 1-channel, 64x64 image tensor for testing."""
    return torch.randn(1, 64, 64)

@pytest.fixture
def sample_image_dict(sample_image_tensor):
    """Provides the sample image tensor within a dictionary."""
    return {"image": sample_image_tensor}

# --- Tests for _normalize_channel ---

@pytest.mark.parametrize("input_array, expected_output", [
    (np.array([0, 50, 100]), np.array([0.0, 0.5, 1.0])),
    (np.array([10, 10, 10]), np.array([0.0, 0.0, 0.0])),
    (np.array([[0, 2], [4, 6]]), np.array([[0, 1/3], [2/3, 1.0]])),
])
def test_normalize_channel(input_array, expected_output):
    """Tests if _normalize_channel correctly scales arrays to the [0, 1] range."""
    normalized = _normalize_channel(input_array)
    assert np.allclose(normalized, expected_output, atol=1e-6)

# --- Tests for WaveletTransformd ---

class TestWaveletTransformd:
    def test_basic_execution(self, sample_image_dict):
        """Tests standard execution with default settings."""
        transform = WaveletTransformd(keys=["image"], output_key="wavelet")
        result = transform(sample_image_dict)
        
        assert "wavelet" in result
        assert isinstance(result["wavelet"], torch.Tensor)
        
        # For a 64x64 image, max_levels is 5. Default is to use max.
        # Channels = 1 (approx) + 3 * 6 (details) = 19.
        assert result["wavelet"].shape == (19, 64, 64)

    def test_with_original_image(self, sample_image_dict):
        """Tests the 'input_original_image' flag."""
        transform = WaveletTransformd(keys=["image"], output_key="wavelet", input_original_image=True, levels=2)
        result = transform(sample_image_dict)
        
        # Channels = 1 (original) + 1 (approx) + 3 * 2 (details) = 8
        assert result["wavelet"].shape[0] == 8
        # The first channel should be the original image
        assert torch.equal(result["wavelet"][0], sample_image_dict["image"].squeeze(0))

    def test_specific_levels(self, sample_image_dict):
        """Tests setting a specific number of decomposition levels."""
        transform = WaveletTransformd(keys=["image"], output_key="wavelet", levels=3)
        result = transform(sample_image_dict)
        # Channels = 1 (approx) + 3 * 3 (details) = 10
        assert result["wavelet"].shape == (10, 64, 64)

    def test_thresholding(self, sample_image_dict):
        """Ensures thresholding runs and alters the output."""
        transform_no_thresh = WaveletTransformd(keys=["image"], output_key="wavelet", levels=2)
        transform_with_thresh = WaveletTransformd(keys=["image"], output_key="wavelet", levels=2, threshold_ratio=0.8)
        
        result_no_thresh = transform_no_thresh(sample_image_dict)
        result_with_thresh = transform_with_thresh(sample_image_dict)
        
        # The outputs should not be identical if thresholding was applied
        assert not torch.equal(result_no_thresh["wavelet"], result_with_thresh["wavelet"])

# --- Tests for RidgeletTransformd ---

@patch('src.data.transforms.FRT')
class TestRidgeletTransformd:
    def test_reconstruction_mode(self, mock_frt_class, sample_image_dict):
        """Tests the default 'reconstruction' output type by mocking the FRT class."""
        # Setup mock FRT instance and its methods
        mock_frt_instance = MagicMock()
        mock_frt_instance.forward.return_value = "coeffs"
        mock_frt_instance.inverse.return_value = np.random.rand(64, 64)
        mock_frt_class.return_value = mock_frt_instance

        transform = RidgeletTransformd(keys=["image"], output_type='reconstruction')
        result = transform(sample_image_dict)

        mock_frt_instance.forward.assert_called_once()
        mock_frt_instance.inverse.assert_called_once_with("coeffs", (64, 64))
        
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (1, 64, 64) # Should have channel dim
        assert result["image"].min() >= 0.0 and result["image"].max() <= 1.0

    def test_sinogram_mode(self, mock_frt_class, sample_image_dict):
        """Tests the 'sinogram' output type."""
        mock_frt_instance = MagicMock()
        mock_frt_instance.forward.return_value = "coeffs"
        mock_frt_instance.reconstruct_sinogram.return_value = np.random.rand(64, 64)
        mock_frt_class.return_value = mock_frt_instance

        transform = RidgeletTransformd(keys=["image"], output_type='sinogram')
        transform(sample_image_dict)

        mock_frt_instance.reconstruct_sinogram.assert_called_once_with("coeffs")

    def test_invalid_output_type(self, mock_frt_class):
        """Tests that an invalid 'output_type' raises a ValueError."""
        with pytest.raises(ValueError, match="output_type must be 'reconstruction' or 'sinogram'"):
            RidgeletTransformd(keys=["image"], output_type='invalid_mode')

# --- Tests for a_trous_decomposition ---

class TestATrousDecomposition:
    def test_output_structure_and_reconstruction(self):
        """Tests the output format and if the layers sum back to the original."""
        image = np.random.rand(32, 32)
        scales = 4
        
        layers = a_trous_decomposition(image, scales=scales)
        
        # Output should be a list of (scales + 1) numpy arrays
        assert isinstance(layers, list)
        assert len(layers) == scales + 1
        assert all(isinstance(layer, np.ndarray) for layer in layers)
        assert all(layer.shape == image.shape for layer in layers)
        
        # The sum of all layers should reconstruct the original image
        reconstructed_image = np.sum(layers, axis=0)
        assert np.allclose(image, reconstructed_image)

    def test_input_validation(self):
        """Tests that the function raises errors for invalid inputs."""
        with pytest.raises(TypeError, match="Input must be a 2D NumPy array."):
            a_trous_decomposition([[1, 2], [3, 4]], scales=2) # Not a numpy array
        
        with pytest.raises(ValueError, match="Scales must be a positive integer."):
            a_trous_decomposition(np.random.rand(32, 32), scales=0)