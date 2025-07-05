import torch
import torch.nn as nn
from torchvision import models

class FractureDetector(nn.Module):
    """
    A ResNet-based model adapted for grayscale (1-channel) fracture detection.

    This class takes a standard torchvision ResNet model, modifies its first
    convolutional layer to accept single-channel images, and replaces its
    final classification layer for a binary output.
    """
    def __init__(self, base_model_name: str = 'resnet18'):
        """
        Initializes the model.

        Args:
            base_model_name (str): The name of the ResNet model to use as a base.
                                   Supported: 'resnet18', 'resnet34', 'resnet50'.
        """
        super().__init__()

        # 1. Load the specified pre-trained ResNet model
        if base_model_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            self.base_model = models.resnet18(weights=weights)
        elif base_model_name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT
            self.base_model = models.resnet34(weights=weights)
        elif base_model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            self.base_model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")

        # 2. Modify the first convolutional layer for 1-channel input
        # Get the original first layer
        original_conv1 = self.base_model.conv1
        
        # Create a new conv layer with 1 input channel
        self.base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Adapt the pre-trained weights from 3 channels to 1 channel
        # We do this by averaging the weights of the original 3 input channels
        self.base_model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # 3. Modify the final classification layer for binary output
        # Get the number of input features for the final layer
        num_features = self.base_model.fc.in_features
        
        # Replace the final layer with a new one for binary classification (1 output)
        self.base_model.fc = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (N, 1, H, W).

        Returns:
            torch.Tensor: The output tensor of shape (N, 1).
        """
        return self.base_model(x)