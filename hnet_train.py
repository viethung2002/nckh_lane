import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import glob
import argparse

from dataloader.hnet_data_processor import HNetDataset
from model.lanenet.Hnet import HNet

# Khởi tạo parser để nhận các đối số từ dòng lệnh
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, help='The phase is train or pretrain')
parser.add_argument('--pre_hnet_weights', type=str, help='The pre hnet weights path')
parser.add_argument('--hnet_weights', type=str, help='The hnet model weights path')
parser.add_argument('--data_path', type=str, help='Path to the gt_label_pts.txt file')
parser.add_argument('--image_dir', type=str, help='Path to the image directory')
args = parser.parse_args()

# Định nghĩa các tham số huấn luyện
batch_size = 1  # Giảm batch size để giảm bộ nhớ GPU
learning_rate_pretrain = 0.0001
learning_rate_train = 0.00005
num_epochs = 20005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình và các thành phần huấn luyện
net = HNet(is_training=True).to(device)
criterion = nn.MSELoss()

# Optimizers cho pretrain và train
pre_optimizer = optim.Adam(net.parameters(), lr=learning_rate_pretrain)
optimizer = optim.Adam(net.parameters(), lr=learning_rate_train)

# Khởi tạo dataset và dataloader
train_dataset = HNetDataset(args.data_path, args.image_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Hàm lưu mô hình
def save_model(model, path, epoch):
    torch.save(model.state_dict(), f"{path}_epoch_{epoch}.pth")

# Hàm áp dụng homography lên các điểm
def apply_homography(points, coefficients):
    H = torch.zeros((3, 3), device=coefficients.device)
    H[0, 0], H[0, 1], H[0, 2] = coefficients[0], coefficients[1], coefficients[2]
    H[1, 0], H[1, 1], H[1, 2] = coefficients[3], coefficients[4], coefficients[5]
    H[2, 0], H[2, 1] = coefficients[6], coefficients[7]
    H[2, 2] = 1.0

    num_points = points.shape[0]
    homogeneous_points = torch.cat([points, torch.ones((num_points, 1), device=points.device)], dim=1)
    
    transformed_points = torch.matmul(homogeneous_points, H.t())
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2].unsqueeze(1)
    return transformed_points

# Chức năng chính để huấn luyện mô hình
def train_hnet(phase, pre_hnet_weights=None, hnet_weights=None):
    if phase == 'pretrain':
        print('Start pretrain hnet......')
        if pre_hnet_weights:
            net.load_state_dict(torch.load(pre_hnet_weights))
            print('Loaded pre hnet weights...')

        for epoch in range(num_epochs):
            net.train()
            for images, gt_points in train_loader:
                images, gt_points = images.to(device), gt_points.to(device)

                # Zero the gradients
                pre_optimizer.zero_grad()

                # Forward pass
                output = net(images)
                
                # Compute loss
                transformed_points = apply_homography(gt_points.squeeze(0), output.squeeze(0))
                pre_loss = criterion(transformed_points, gt_points.squeeze(0))

                # Backward pass and optimize
                pre_loss.backward()
                pre_optimizer.step()

            if epoch % 100 == 0:
                print(f'[{epoch}] pretrain hnet pre loss = {pre_loss.item()}')

            if epoch % 1000 == 0:
                save_model(net, './model/hnet', epoch)

    elif phase == 'train':
        print('Start train hnet......')
        if hnet_weights:
            net.load_state_dict(torch.load(hnet_weights))
            print('Loaded hnet weights...')
        elif pre_hnet_weights:
            net.load_state_dict(torch.load(pre_hnet_weights))
            print('Loaded pre hnet weights...')
        else:
            print('Train from scratch without H matrix initialize.')

        for epoch in range(num_epochs):
            net.train()
            for images, gt_points in train_loader:
                images, gt_points = images.to(device), gt_points.to(device)

                optimizer.zero_grad()

                output = net(images)
                transformed_points = apply_homography(gt_points.squeeze(0), output.squeeze(0))
                loss = criterion(transformed_points, gt_points.squeeze(0))

                loss.backward()
                optimizer.step()

            if epoch % 50 == 0:
                print(f'epoch[{epoch}], hnet training loss = {loss.item()}')

            if epoch % 1000 == 0:
                save_model(net, './model/hnet', epoch)

    else:
        print('Wrong phase!!!!!!')

# Khởi động chương trình huấn luyện
if __name__ == '__main__':
    train_hnet(args.phase, args.pre_hnet_weights, args.hnet_weights)
