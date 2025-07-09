# src/models/model.py

import torch
import torch.nn as nn
from torchvision import models

class PathologyDetector(nn.Module):
    """
    A factory model that can build either a ResNet-based or a Vision 
    Transformer-based model, adapted for grayscale or multi-channel inputs.

    This class takes a standard torchvision model, modifies its first
    input layer to accept `in_channels` and replaces its final
    classification layer for a binary output.
    """
    def __init__(self, base_model_name: str = 'resnet18', in_channels: int = 1):
        """
        Initializes the model.

        Args:
            base_model_name (str): The name of the model to use as a base.
                Supported ResNet: 'resnet18', 'resnet34', 'resnet50'.
                Supported ViT: 'vit_b_16', 'vit_b_32'.
            in_channels (int): The number of input channels for the model.
        """
        super().__init__()

        # --- Model Loading ---
        if 'resnet' in base_model_name:
            self.base_model = self._create_resnet(base_model_name, in_channels)
        elif 'vit' in base_model_name:
            self.base_model = self._create_vit(base_model_name, in_channels)
        else:
            raise ValueError(f"Unsupported base model family: {base_model_name}")

    def _create_resnet(self, base_model_name, in_channels):
        """Builds and modifies a ResNet model."""
        # 1. Load the specified pre-trained ResNet model
        if base_model_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        elif base_model_name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT
            model = models.resnet34(weights=weights)
        elif base_model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet model: {base_model_name}")

        # 2. Modify the first convolutional layer for `in_channels` input
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # Adapt pre-trained weights
        averaged_weights = original_conv1.weight.data.mean(dim=1, keepdim=True)
        model.conv1.weight.data = averaged_weights.repeat(1, in_channels, 1, 1)

        # 3. Modify the final classification layer for binary output
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        
        return model

    def _create_vit(self, base_model_name, in_channels):
        """Builds and modifies a Vision Transformer model."""
        # 1. Load the specified pre-trained ViT model
        if base_model_name == 'vit_b_16':
            weights = models.ViT_B_16_Weights.DEFAULT
            model = models.vit_b_16(weights=weights)
        elif base_model_name == 'vit_b_32':
            weights = models.ViT_B_32_Weights.DEFAULT
            model = models.vit_b_32(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT model: {base_model_name}")

        # 2. Modify the first layer (patch embedding) for `in_channels` input
        original_conv_proj = model.conv_proj
        model.conv_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv_proj.out_channels,
            kernel_size=original_conv_proj.kernel_size,
            stride=original_conv_proj.stride,
            padding=original_conv_proj.padding,
            bias=(original_conv_proj.bias is not None)
        )
        
        # NOTE: The weight adaptation logic is deliberately omitted.
        # This new `conv_proj` layer will be trained from scratch with random
        # initialization, allowing it to learn features for the novel
        # multi-channel (wavelet) input while the rest of the model
        # remains pretrained.

        # 3. Modify the final classification layer (head) for binary output
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, 1)
        
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        return self.base_model(x)