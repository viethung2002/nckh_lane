import torch
import torch.nn as nn

class HNet(nn.Module):
    def __init__(self, is_training=True):
        super(HNet, self).__init__()
        self.is_training = is_training

        # Mạng CNN để trích xuất đặc trưng
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Điều chỉnh kích thước dựa trên đầu ra từ các lớp convolution
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 16, 1024),  # 64 * 8 * 16 = 8192
            nn.ReLU(inplace=True),
            nn.Linear(1024, 8)  # Dự đoán 8 hệ số cho homography
        )

    def forward(self, x):
        x = self.features(x)
        print(f"Shape after conv layers: {x.shape}")
        
        x = x.view(x.size(0), -1)
        print(f"Shape after flatten: {x.shape}")
        
        x = self.fc(x)
        return x
