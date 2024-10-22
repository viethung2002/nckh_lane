import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18  # Import backbone

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SpatialPath(nn.Module):
    def __init__(self, in_ch):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64, stride=2)
        self.conv2 = ConvBlock(64, 128, stride=2)
        self.conv3 = ConvBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class ResNet18_Extract(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18_Extract, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f3, f4  # Trả về đặc trưng từ layer3 và layer4

class ContextPath(nn.Module):
    def __init__(self, backbone):
        super(ContextPath, self).__init__()
        self.backbone = backbone
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.arm1 = nn.Conv2d(512, 128, kernel_size=1)
        self.arm2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_gc = nn.Conv2d(512, 128, kernel_size=1)  # Thêm lớp này

    def forward(self, x):
        feat3, feat4 = self.backbone(x)
        # Global Average Pooling
        global_context = self.global_avg_pool(feat4)
        global_context = F.interpolate(global_context, size=feat4.size()[2:], mode='bilinear', align_corners=True)
        global_context = self.conv_gc(global_context)  # Áp dụng convolution để giảm số kênh
        feat4 = self.arm1(feat4) + global_context
        feat4 = F.interpolate(feat4, size=feat3.size()[2:], mode='bilinear', align_corners=True)
        feat3 = self.arm2(feat3) + feat4
        return feat3  # Trả về đặc trưng kết hợp

class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FeatureFusionModule, self).__init__()
        self.conv_block = ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, sp, cp):
        cp = F.interpolate(cp, size=sp.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([sp, cp], dim=1)
        x = self.conv_block(x)
        x = self.conv1x1(x)
        return self.relu(x)

class BiSeNet_Encoder(nn.Module):
    def __init__(self, in_ch, backbone):
        super(BiSeNet_Encoder, self).__init__()
        self.spatial_path = SpatialPath(in_ch)
        self.context_path = ContextPath(backbone)

    def forward(self, x):
        sp = self.spatial_path(x)
        cp = self.context_path(x)
        return sp, cp

class BiSeNet_Decoder(nn.Module):
    def __init__(self, out_ch):
        super(BiSeNet_Decoder, self).__init__()
        # Tổng số kênh đầu vào là 256 (sp) + 128 (cp) = 384
        self.feature_fusion = FeatureFusionModule(in_ch=384, out_ch=256)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, sp, cp):
        x = self.feature_fusion(sp, cp)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        return x
