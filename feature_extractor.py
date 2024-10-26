import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CustomResNet18(nn.Module):
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        super(CustomResNet18, self).__init__()
        # Load the pre-trained ResNet-18 model
        self.resnet18 = resnet18(weights=weights, progress= False)
        
        # Remove the final fully connected (classifier) layer and avgpool layer
        self.feature_extractor = nn.Sequential(*list(self.resnet18.children())[:-2])
        
        # Add a Conv2d layer to adjust the output channels to 1
        self.conv_adjust = nn.Conv2d(512, 1, kernel_size=1)  # Change number of output channels to 1
        
        # Add an Upsampling layer to resize to (128, 128)
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)


        for param in list(self.feature_extractor.children())[:10]:
            param.requires_grad = False

    def forward(self, x):
        # Forward pass through the feature extractor
        features = self.feature_extractor(x)  # Output shape: (batch_size, 512, 7, 7)
        
        # Adjust the output channels to 1
        features = self.conv_adjust(features)  # Output shape: (batch_size, 1, 7, 7)
        
        # Upsample to (128, 128)
        features = self.upsample(features)  # Output shape: (batch_size, 1, 128, 128)
        
        return features

# # Example usage:
# model = CustomResNet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# input_tensor = torch.randn(64, 3, 224, 224)  # Batch size of 4, 3 channels, 224x224 image
# output_features = model(input_tensor)
# print("Output feature map shape:", output_features.shape)  # Should be (4, 1, 128, 128)
