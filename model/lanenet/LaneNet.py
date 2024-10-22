# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lanenet.loss import DiscriminativeLoss
from model.lanenet.backbone.UNet import UNet_Encoder, UNet_Decoder
from model.lanenet.backbone.ENet import ENet_Encoder, ENet_Decoder
from model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder, Deeplabv3plus_Decoder
from model.lanenet.backbone.LinkNet import LinkNet_Encoder, LinkNet_Decoder  # Import LinkNet
from model.lanenet.backbone.BiSeNet import BiSeNet_Encoder, BiSeNet_Decoder, ResNet18_Extract  # Import BiSeNet và ResNet18_Extract
from model.lanenet.backbone.ERFNet import ERFNet_Encoder, ERFNet_Decoder  # Import ERFNet

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LaneNet(nn.Module):
    def __init__(self, in_ch=3, arch="ENet"):
        super(LaneNet, self).__init__()
        # Số lượng instance cho segmentation
        self.no_of_instances = 3  # Nếu bạn muốn đầu ra là bản đồ instance RGB, nên đặt là 3.
        print("Use {} as backbone".format(arch))
        self._arch = arch
        if self._arch == 'UNet':
            self._encoder = UNet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = UNet_Decoder(2)
            self._decoder_instance = UNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENet':
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = ENet_Decoder(2)
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'DeepLabv3+':
            self._encoder = Deeplabv3plus_Encoder()
            self._encoder.to(DEVICE)

            self._decoder_binary = Deeplabv3plus_Decoder(2)
            self._decoder_instance = Deeplabv3plus_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'LinkNet':
            self._encoder = LinkNet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = LinkNet_Decoder(2)
            self._decoder_instance = LinkNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'BiSeNet':
            # Khởi tạo backbone
            backbone = ResNet18_Extract(pretrained=True)
            backbone.to(DEVICE)
            self._encoder = BiSeNet_Encoder(in_ch, backbone=backbone)
            self._encoder.to(DEVICE)

            self._decoder_binary = BiSeNet_Decoder(2)
            self._decoder_instance = BiSeNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ERFNet':
            self._encoder = ERFNet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = ERFNet_Decoder(2)
            self._decoder_instance = ERFNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        else:
            raise ValueError("Please select a valid model.")

        self.relu = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def forward(self, input_tensor):
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2, c3, c4, c5)
            instance = self._decoder_instance(c1, c2, c3, c4, c5)
        elif self._arch == 'ENet':
            c = self._encoder(input_tensor)
            binary = self._decoder_binary(c)
            instance = self._decoder_instance(c)
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2)
            instance = self._decoder_instance(c1, c2)
        elif self._arch == 'LinkNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2, c3, c4, c5)
            instance = self._decoder_instance(c1, c2, c3, c4, c5)
        elif self._arch == 'BiSeNet':
            sp, cp = self._encoder(input_tensor)
            binary = self._decoder_binary(sp, cp)
            instance = self._decoder_instance(sp, cp)
        elif self._arch == 'ERFNet':
            c = self._encoder(input_tensor)
            binary = self._decoder_binary(c)
            instance = self._decoder_instance(c)
        else:
            raise ValueError("Please select a valid model.")

        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)

        pix_embedding = self.sigmoid(instance)

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }
