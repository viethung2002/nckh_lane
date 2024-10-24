import torch
import torch.nn as nn
from model.scnn.backbone.SCNN import SCNN  # Import mô hình SCNN từ backbone

class LaneDetectionWithSCNN(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        """
        Khởi tạo mô hình Lane Detection với backbone là SCNN.
        - in_ch: số lượng kênh đầu vào (thường là 3 kênh RGB).
        - out_ch: số lượng kênh đầu ra (thường là 2 kênh cho phân đoạn nhị phân).
        """
        super(LaneDetectionWithSCNN, self).__init__()

        # Thay vì sử dụng các backbone khác, sử dụng SCNN trực tiếp
        self.scnn = SCNN(in_channels=in_ch, out_channels=out_ch)

        # Lớp phân đoạn nhị phân
        self.classifier = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """
        Forward pass qua SCNN.
        - x: ảnh đầu vào.
        - Trả về: đầu ra của phân đoạn nhị phân cho ảnh.
        """
        # Truyền qua SCNN
        spatial_features = self.scnn(x)

        # Kiểm tra kích thước sau SCNN
        print(f"Shape of spatial_features from SCNN: {spatial_features.shape}")

        # Phân đoạn nhị phân
        output = self.classifier(spatial_features)

        return output
