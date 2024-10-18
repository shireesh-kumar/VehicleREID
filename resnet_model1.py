import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models

# Define the Bottleneck class
class Bottleneck(nn.Module):
    def __init__(self, in_channels, intermediate_channels, expansion, is_Bottleneck, stride):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck
        
        if self.in_channels == self.intermediate_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels * self.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.intermediate_channels * self.expansion)
            )

        self.relu = nn.ReLU()

        if self.is_Bottleneck:
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.intermediate_channels)
            self.conv2 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.intermediate_channels)
            self.conv3 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.intermediate_channels * self.expansion)
        else:
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.intermediate_channels)
            self.conv2 = nn.Conv2d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.intermediate_channels)

    def forward(self, x):
        in_x = x
        if self.is_Bottleneck:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))

        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)

        return self.relu(x)

# Define the ResNet class
class ResNet(nn.Module):
    def __init__(self, resnet_variant, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_blocks(64, self.channels_list[0], self.repeatition_list[0], self.expansion, self.is_Bottleneck, stride=1)
        self.block2 = self._make_blocks(self.channels_list[0] * self.expansion, self.channels_list[1], self.repeatition_list[1], self.expansion, self.is_Bottleneck, stride=2)
        self.block3 = self._make_blocks(self.channels_list[1] * self.expansion, self.channels_list[2], self.repeatition_list[2], self.expansion, self.is_Bottleneck, stride=2)
        self.block4 = self._make_blocks(self.channels_list[2] * self.expansion, self.channels_list[3], self.repeatition_list[3], self.expansion, self.is_Bottleneck, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def _make_blocks(self, in_channels, intermediate_channels, num_repeat, expansion, is_Bottleneck, stride):
        layers = []
        layers.append(Bottleneck(in_channels, intermediate_channels, expansion, is_Bottleneck, stride=stride))
        for _ in range(1, num_repeat):
            layers.append(Bottleneck(intermediate_channels * expansion, intermediate_channels, expansion, is_Bottleneck, stride=1))
        return nn.Sequential(*layers)



# # Function to test the ResNet model
# def test_ResNet(params):
#     model = ResNet(params['resnet50'], in_channels=3, num_classes=1000)  # Change num_classes as needed

#     # Load pre-trained ImageNet weights
#     pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

#     # Load the state_dict into your model
#     model.load_state_dict(pretrained_model.state_dict(), strict=True)

#     # Move model to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # Testing the model with a random input
#     x = torch.randn(1, 3, 224, 224).to(device)  # Move input to the same device as model
#     output = model(x)
#     print(output.shape)
#     return model

# # Parameters for ResNet50
# model_parameters = {
#     'resnet50': ([64, 128, 256, 512], [3, 4, 6, 3], 4, True),
#     'resnet34' : ([64, 128, 256, 512 ], [3, 4, 6, 3 ], 1 ,  False)
# }
# # Test the ResNet model
# model = test_ResNet(model_parameters)

# # Print the model summary
# summary(model, (3, 224, 224))
