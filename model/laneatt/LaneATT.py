import torch
import torch.nn as nn
import torch.nn.functional as F
from model.laneatt.backbone.LaneATTBackbone import LaneATTBackbone  # Import LaneATT backbone

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LaneATT(nn.Module):
    def __init__(self, in_ch=3, backbone='resnet50', pretrained=True):
        super(LaneATT, self).__init__()
        # Number of instances for segmentation
        self.no_of_instances = 3  # If you want the output to be an RGB instance map, set to 3.
        print("Using {} as backbone".format(backbone))
        self._backbone_type = backbone
        
        # Initialize the LaneATT Backbone
        self._encoder = LaneATTBackbone(backbone=self._backbone_type, pretrained=pretrained)
        self._encoder.to(DEVICE)

        # Determine number of channels based on backbone type
        if backbone == 'resnet50':
            encoder_out_channels = 128  # Adjusted based on actual output from encoder
        elif backbone == 'resnet34' or backbone == 'resnet18':
            encoder_out_channels = 128  # Adjusted based on actual output from encoder
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Additional layers for binary and instance segmentation
        self.conv1x1 = nn.Conv2d(encoder_out_channels, 512, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv_binary = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)
        self.conv_instance = nn.Conv2d(128, self.no_of_instances, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

        # Upsample to match input size
        self.upsample = nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)  # Ensure output matches input size

    def forward(self, input_tensor):
        # Feature extraction using LaneATT backbone
        features = self._encoder(input_tensor)

        # Check the output shape of the encoder
        # print(f"Encoder output shape: {features.shape}")

        # Additional convolution layers for lane feature extraction
        features = self.conv1x1(features)
        features = self.relu(features)
        features = self.conv3x3(features)
        features = self.relu(features)
        features = self.dropout(features)

        # Binary segmentation head
        binary = self.conv_binary(features)

        # Instance segmentation head
        instance = self.conv_instance(features)

        # Upsample to match input size
        binary = self.upsample(binary)
        instance = self.upsample(instance)

        # # Check the upsampled output shapes
        # print(f"Upsampled binary output shape: {binary.shape}")
        # print(f"Upsampled instance output shape: {instance.shape}")

        # Generate binary segmentation prediction
        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)

        # Apply sigmoid to the instance segmentation output
        pix_embedding = self.sigmoid(instance)

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }

if __name__ == "__main__":
    # Example to test the LaneATT model
    model = LaneATT(backbone='resnet50', pretrained=False)
    input_tensor = torch.randn(1, 3, 256, 512).to(DEVICE)  # Example input with batch size of 1 and image size 256x512
    output = model(input_tensor)
    print(f"Instance segmentation logits shape: {output['instance_seg_logits'].shape}")
    print(f"Binary segmentation prediction shape: {output['binary_seg_pred'].shape}")
    print(f"Binary segmentation logits shape: {output['binary_seg_logits'].shape}")

