import torch
import torch.nn as nn
import torch.nn.functional as F

class SCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(SCNN, self).__init__()

        # Layer convolution đầu vào
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Spatial Convolution - SCNN-specific layer
        self.spatial_conv = nn.Conv2d(256, 256, kernel_size=(1, 9), stride=1, padding=(0, 4))

        # Decoder - chuyển từ biểu diễn không gian thành dự đoán nhị phân (binary segmentation)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Spatial Convolution (SCNN-specific)
        x = self.relu(self.spatial_conv(x))

        # Decoder
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)

        return self.sigmoid(x)
