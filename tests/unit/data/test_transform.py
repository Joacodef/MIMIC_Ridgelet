# tests/unit/data/test_transforms.py
import torch
import numpy as np
import pytest
from src.data.transforms import RidgeletTransformd

class TestRidgeletTransformd:
    """
    Unit tests for the RidgeletTransformd class.
    """

    @pytest.mark.parametrize("size", [128, 256])
    def test_ridgelet_transform_output_shape_and_type(self, size):
        """
        Tests that the RidgeletTransformd returns a tensor of the same shape
        and correct type for various valid input sizes.
        """
        # 1. Setup
        image_size = (1, 1, size, size)
        input_tensor = torch.randn(image_size, dtype=torch.float32)
        data = {"image": input_tensor}
        transform = RidgeletTransformd(keys=["image"])

        # 2. Execution
        result = transform(data)

        # 3. Assertion
        output_tensor = result["image"]
        assert isinstance(output_tensor, torch.Tensor)
        # The output shape will have a single channel
        assert output_tensor.shape == (1, size, size)
        assert output_tensor.dtype == torch.float32
        # Ensure the transform actually changed the image
        assert not torch.equal(input_tensor.squeeze(0), output_tensor)

    def test_transform_raises_error_for_nonsquare_image(self):
        """
        Tests that the transform correctly raises a ValueError when the
        input image is not square.
        """
        # 1. Setup
        image_size = (1, 1, 256, 128) # Non-square
        input_tensor = torch.randn(image_size, dtype=torch.float32)
        data = {"image": input_tensor}
        transform = RidgeletTransformd(keys=["image"])

        # 2. Execution and Assertion
        with pytest.raises(ValueError, match="Image must be a 2D square array for the FRT"):
            transform(data)

    def test_transform_raises_error_for_invalid_type(self):
        """
        Tests that the transform correctly raises a TypeError when the
        input is not a torch.Tensor.
        """
        # 1. Setup
        input_numpy = np.random.randn(1, 256, 256).astype(np.float32)
        data = {"image": input_numpy}
        transform = RidgeletTransformd(keys=["image"])

        # 2. Execution and Assertion
        with pytest.raises(TypeError, match="Input must be a PyTorch tensor"):
            transform(data)