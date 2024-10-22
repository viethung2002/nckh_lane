import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(noutput)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = torch.cat([self.conv(input), self.pool(input)], 1)
        x = self.bn(x)
        x = self.relu(x)
        return x

class NonBottleneck1D(nn.Module):
    def __init__(self, chann, dropprob, dilated):       
        super(NonBottleneck1D, self).__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3,1), padding=(1,0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), padding=(0,1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3,1), padding=(dilated,0), bias=True, dilation=(dilated,1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), padding=(0,dilated), bias=True, dilation=(1,dilated))
        self.bn2 = nn.BatchNorm2d(chann)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)
    
    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)
        
        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        
        if self.dropout.p != 0:
            output = self.dropout(output)
        
        return self.relu(output + input)  # +input for residual connection

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return self.relu(output)

class ERFNet_Encoder(nn.Module):
    def __init__(self, in_ch):
        super(ERFNet_Encoder, self).__init__()
        self.initial_block = DownsamplerBlock(in_ch, 16)
        
        self.down1 = DownsamplerBlock(16, 64)
        self.non_bottleneck1 = nn.Sequential(
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1)
        )
        
        self.down2 = DownsamplerBlock(64, 128)
        self.non_bottleneck2 = nn.Sequential(
            NonBottleneck1D(128, 0.3, 2),
            NonBottleneck1D(128, 0.3, 4),
            NonBottleneck1D(128, 0.3, 8),
            NonBottleneck1D(128, 0.3, 16),
            NonBottleneck1D(128, 0.3, 2),
            NonBottleneck1D(128, 0.3, 4),
            NonBottleneck1D(128, 0.3, 8),
            NonBottleneck1D(128, 0.3, 16)
        )

    def forward(self, input):
        output = self.initial_block(input)
        output = self.down1(output)
        output = self.non_bottleneck1(output)
        output = self.down2(output)
        output = self.non_bottleneck2(output)
        return output  # Trả về đặc trưng có số kênh là 128

class ERFNet_Decoder(nn.Module):
    def __init__(self, out_ch):
        super(ERFNet_Decoder, self).__init__()
        self.up1 = UpsamplerBlock(128, 64)
        self.non_bottleneck1 = nn.Sequential(
            NonBottleneck1D(64, 0, 1),
            NonBottleneck1D(64, 0, 1)
        )
        self.up2 = UpsamplerBlock(64, 16)
        self.non_bottleneck2 = nn.Sequential(
            NonBottleneck1D(16, 0, 1),
            NonBottleneck1D(16, 0, 1)
        )
        self.output_conv = nn.ConvTranspose2d(16, out_ch, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.up1(input)
        output = self.non_bottleneck1(output)
        output = self.up2(output)
        output = self.non_bottleneck2(output)
        output = self.output_conv(output)
        return output
