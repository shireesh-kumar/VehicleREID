import torch
import torch.nn as nn
from torchvision import models

class CustomResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomResNet34, self).__init__()
        # Load the pre-trained ResNet-34 model
        self.resnet34 = models.resnet34(pretrained=pretrained)
        
        # Remove the fully connected layer (classifier)
        self.feature_extractor = nn.Sequential(*list(self.resnet34.children())[:-2])
        
        # Add a Conv2d layer to adjust the output channels to 1
        self.conv_adjust = nn.Conv2d(512, 1, kernel_size=1)  # Change number of output channels to 1
        
        # Add an Upsampling layer to resize to (128, 128)
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Forward pass through the feature extractor
        features = self.feature_extractor(x)
        
        # Adjust the output size to 128x128 and change to single channel
        features = self.conv_adjust(features)  # Output shape: (batch_size, 1, 7, 7)
        
        # Upsample to (128, 128)
        features = self.upsample(features)  # Output shape: (batch_size, 1, 128, 128)
        
        return features

# # Example usage
# model = CustomResNet34(pretrained=True)

# # Dummy input
# input_tensor = torch.randn(4, 3, 224, 224)  # Batch size of 4, 3 channels, 224x224 image
# output_features = model(input_tensor)

# print("Output feature map shape:", output_features.shape)  # Should be (4, 1, 128, 128)
