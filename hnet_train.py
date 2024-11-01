import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import glob
import argparse
from tqdm import tqdm  # Added import for tqdm

from dataloader.hnet_data_processor import HNetDataset
from model.lanenet.Hnet import HNet

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, help='The phase is train or pretrain')
parser.add_argument('--pre_hnet_weights', type=str, help='The pre hnet weights path')
parser.add_argument('--hnet_weights', type=str, help='The hnet model weights path')
parser.add_argument('--data_path', type=str, help='Path to the gt_label_pts.txt file')
parser.add_argument('--image_dir', type=str, help='Path to the image directory')
args = parser.parse_args()

# Define training parameters
batch_size = 16  # Reduce batch size to lower GPU memory usage
learning_rate_pretrain = 0.0001
learning_rate_train = 0.00005
num_epochs = 20005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and training components
net = HNet(is_training=True).to(device)
criterion = nn.MSELoss()

# Optimizers for pretrain and train
pre_optimizer = optim.Adam(net.parameters(), lr=learning_rate_pretrain)
optimizer = optim.Adam(net.parameters(), lr=learning_rate_train)

# Initialize dataset and dataloader
train_dataset = HNetDataset(args.data_path, args.image_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Function to save the model
def save_model(model, path, epoch):
    torch.save(model.state_dict(), f"{path}_epoch_{epoch}.pth")

# Function to apply homography to points
def apply_homography(points, coefficients):
    # Ensure points have three dimensions
    if points.dim() == 2:
        points = points.unsqueeze(1)  # Add num_points dimension

    batch_size, num_points, _ = points.size()
    
    # Initialize homography matrices
    H = torch.zeros((batch_size, 3, 3), device=coefficients.device)
    H[:, 0, 0] = coefficients[:, 0]
    H[:, 0, 1] = coefficients[:, 1]
    H[:, 0, 2] = coefficients[:, 2]
    H[:, 1, 0] = coefficients[:, 3]
    H[:, 1, 1] = coefficients[:, 4]
    H[:, 1, 2] = coefficients[:, 5]
    H[:, 2, 0] = coefficients[:, 6]
    H[:, 2, 1] = coefficients[:, 7]
    H[:, 2, 2] = 1.0

    # Create homogeneous coordinates
    homogeneous_points = torch.cat([
        points, 
        torch.ones((batch_size, num_points, 1), device=points.device)
    ], dim=-1)  # Shape: [batch_size, num_points, 3]
    
    # Apply homography
    transformed_points = torch.bmm(homogeneous_points, H.transpose(1, 2))  # [batch_size, 3, num_points]
    transformed_points = transformed_points[:, :, :2] / transformed_points[:, :, 2].unsqueeze(-1)  # [batch_size, 2, num_points]
    
    # Permute to [batch_size, num_points, 2]
    transformed_points = transformed_points.permute(0, 2, 1)
    
    return transformed_points

# Main training function
def train_hnet(phase, pre_hnet_weights=None, hnet_weights=None):
    if phase == 'pretrain':
        print('Start pretrain hnet......')
        if pre_hnet_weights:
            net.load_state_dict(torch.load(pre_hnet_weights))
            print('Loaded pre hnet weights...')

        for epoch in range(num_epochs):
            net.train()
            epoch_loss = 0.0
            # Initialize tqdm for batches
            with tqdm(total=len(train_loader), desc=f'Pretrain Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for images, gt_points in train_loader:
                    images, gt_points = images.to(device), gt_points.to(device)

                    # Zero the gradients
                    pre_optimizer.zero_grad()

                    # Forward pass
                    output = net(images)
                    
                    # Compute loss for the entire batch
                    transformed_points = apply_homography(gt_points, output)
                    pre_loss = criterion(transformed_points, gt_points)

                    # Backward pass and optimize
                    pre_loss.backward()
                    pre_optimizer.step()

                    epoch_loss += pre_loss.item()
                    pbar.set_postfix({'Loss': pre_loss.item()})
                    pbar.update(1)

            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f'[{epoch+1}] pretrain hnet average loss = {avg_loss}')

            if (epoch + 1) % 1000 == 0:
                save_model(net, './model/hnet', epoch+1)

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
            epoch_loss = 0.0
            # Initialize tqdm for batches
            with tqdm(total=len(train_loader), desc=f'Train Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for images, gt_points in train_loader:
                    images, gt_points = images.to(device), gt_points.to(device)

                    optimizer.zero_grad()

                    output = net(images)
                    transformed_points = apply_homography(gt_points, output)
                    loss = criterion(transformed_points, gt_points)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'Loss': loss.item()})
                    pbar.update(1)

            if (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f'epoch[{epoch+1}], hnet training average loss = {avg_loss}')

            if (epoch + 1) % 1000 == 0:
                save_model(net, './model/hnet', epoch+1)

    else:
        print('Wrong phase!!!!!!')

# Start the training program
if __name__ == '__main__':
    train_hnet(args.phase, args.pre_hnet_weights, args.hnet_weights)
