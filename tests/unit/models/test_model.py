import pytest
import torch
from torchvision import models

# Adjust the import path if your project structure is different
from src.models.model import FractureDetector

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def dummy_input_tensor_factory():
    """A factory to create dummy input tensors of a specific shape."""
    def _create(batch_size, in_channels, height, width):
        return torch.randn(batch_size, in_channels, height, width)
    return _create

# --- Tests for FractureDetector ---

class TestFractureDetector:

    @pytest.mark.parametrize("base_model_name, in_channels", [
        ('resnet18', 1),
        ('resnet34', 4),
        ('resnet50', 19)
    ])
    def test_initialization_structure(self, base_model_name, in_channels):
        """
        Tests if the model initializes with the correct structure:
        - The first conv layer has the correct number of input channels.
        - The final FC layer has a single output neuron.
        """
        # Act
        model = FractureDetector(base_model_name=base_model_name, in_channels=in_channels)
        
        # Assert
        # Check that the first convolutional layer is correctly modified
        first_conv_layer = model.base_model.conv1
        assert isinstance(first_conv_layer, torch.nn.Conv2d)
        assert first_conv_layer.in_channels == in_channels
        
        # Check that the final fully connected layer is correctly modified
        final_fc_layer = model.base_model.fc
        assert isinstance(final_fc_layer, torch.nn.Linear)
        assert final_fc_layer.out_features == 1

    @pytest.mark.parametrize("in_channels, batch_size", [(1, 1), (4, 8)])
    def test_forward_pass(self, in_channels, batch_size, dummy_input_tensor_factory):
        """
        Tests if the forward pass executes without errors and produces an
        output tensor of the expected shape (N, 1).
        """
        # Arrange
        model = FractureDetector(base_model_name='resnet18', in_channels=in_channels)
        model.eval()  # Set to evaluation mode
        dummy_input = dummy_input_tensor_factory(batch_size, in_channels, 224, 224)

        # Act
        with torch.no_grad():
            output = model(dummy_input)

        # Assert
        assert output.shape == (batch_size, 1)

    def test_weight_adaptation_logic(self):
        """
        Tests the specific logic for adapting the pretrained weights of the first
        convolutional layer from 3 channels to a new number of channels.
        """
        # Arrange: Get the original 3-channel weights from a standard ResNet18
        original_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        original_weights = original_model.conv1.weight.data.clone()
        
        # Act: Create our model with 1 input channel
        custom_model = FractureDetector(base_model_name='resnet18', in_channels=1)
        custom_weights = custom_model.base_model.conv1.weight.data
        
        # Assert: Check if the custom weights are the mean of the original weights
        # Calculate the expected weights by averaging across the input channel dim (dim=1)
        expected_weights = original_weights.mean(dim=1, keepdim=True)
        
        assert torch.allclose(custom_weights, expected_weights, atol=1e-6)

    def test_unsupported_model_error(self):
        """
        Tests that the class raises a ValueError when an unsupported
        base_model_name is provided.
        """
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported base model: efficientnet"):
            FractureDetector(base_model_name='efficientnet', in_channels=1)