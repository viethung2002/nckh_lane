import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, resnet34, ResNet50_Weights, ResNet34_Weights

class LaneATTBackbone(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(LaneATTBackbone, self).__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet34':
            self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Adding additional layers for lane detection
        if backbone == 'resnet50':
            in_channels = 2048
        elif backbone == 'resnet34':
            in_channels = 512
        elif backbone == 'resnet18':
            in_channels = 512

        self.conv1x1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Extract features using ResNet backbone
        x = self.backbone(x)
        # Additional convolutions for lane feature extraction
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.conv3x3(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    # Example to test the backbone
    model = LaneATTBackbone(backbone='resnet50', pretrained=False)
    input_tensor = torch.randn(1, 3, 256, 512)  # Example input with batch size of 1 and image size 256x512
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
