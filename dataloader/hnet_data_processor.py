import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class HNetDataset(Dataset):
    def __init__(self, gt_label_pts_file, image_dir):
        self.image_dir = image_dir
        self.data = []

        # Đọc file gt_label_pts.txt
        with open(gt_label_pts_file, 'r') as file:
            for line in file:
                image_name, x, y, label = line.strip().split(',')
                self.data.append((image_name, int(x), int(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, x, y = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load hình ảnh và chuyển đổi thành tensor
        image = Image.open(image_path).convert('RGB')
        image = image.resize((128, 64))  # Resize hình ảnh nếu cần
        image = np.array(image).astype(np.float32) / 255.0  # Chuẩn hóa ảnh
        image = torch.from_numpy(image).permute(2, 0, 1)  # Chuyển đổi sang tensor với thứ tự (C, H, W)

        # Trả về điểm gt (x, y) dưới dạng tensor
        gt_points = torch.tensor([x, y, 1.0], dtype=torch.float32)  # Include homogeneous coordinate

        return image, gt_points
