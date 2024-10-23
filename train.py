import time
import os
import sys

import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from model.csnn.LaneDetectionCSNN import LaneDetectionWithCSNN
from torch.utils.data import DataLoader
from torchvision import transforms
from model.utils.cli_helper import parse_args
from model.utils.train_utils import train_lanenet_model, train_csnn_model  # Import các hàm huấn luyện từ train_utils.py
import pandas as pd
from tqdm import tqdm

from model.lanenet.loss import compute_loss  # Import trực tiếp hàm compute_loss từ loss.py

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    args = parse_args()
    save_path = args.save
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    resize_height = args.height
    resize_width = args.width

    # Định nghĩa các phép biến đổi dữ liệu
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    # Tạo dataset và dataloaders
    train_dataset = TusimpleSet(train_dataset_file, transform=data_transforms['train'], target_transform=target_transforms)
    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'], target_transform=target_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.bs, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=args.bs, shuffle=True)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Khởi tạo mô hình dựa trên `model_type`
    if args.model_type.lower() == 'lanenet':
        model = LaneNet(arch=args.backbone)
        train_fn = train_lanenet_model
    elif args.model_type.lower() == 'csnn':
        model = LaneDetectionWithCSNN()
        train_fn = train_csnn_model
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.to(DEVICE)

    # Khởi tạo optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize start_epoch
    start_epoch = 0

    # Load model_state_dict nếu có
    if args.pretrained_model is not None:
        if os.path.isfile(args.pretrained_model):
            print('Loading model state_dict from {}'.format(args.pretrained_model))
            model_state = torch.load(args.pretrained_model, map_location=DEVICE)
            model.load_state_dict(model_state)
            print('Model state_dict loaded successfully.')
        else:
            print('Model state_dict not found at {}'.format(args.pretrained_model))
            sys.exit(1)

    print(f"Starting training from epoch {start_epoch} for {args.epochs} epochs with {len(train_dataset)} training samples\n")

    # Biến theo dõi loss tốt nhất
    best_val_loss = float('inf')
    best_model_wts = None

    # Gọi hàm huấn luyện tương ứng với mô hình
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch}/{args.epochs - 1}')
        print('-' * 10)

        # Mỗi epoch sẽ có hai phase: train và val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Đặt mô hình vào chế độ huấn luyện
            else:
                model.eval()  # Đặt mô hình vào chế độ đánh giá

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0
            correct_binary = 0
            total_pixels = 0
            false_positive_result = 0
            true_positive_result = 0

            # Thêm thanh tiến trình cho mỗi epoch
            with tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch}') as pbar:
                for inputs, binarys, instances in dataloaders[phase]:
                    inputs = inputs.type(torch.FloatTensor).to(DEVICE)
                    binarys = binarys.type(torch.LongTensor).to(DEVICE)
                    instances = instances.type(torch.FloatTensor).to(DEVICE)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        total_loss, binary_loss, instance_loss, _ = compute_loss(outputs, binarys, instances, args.loss_type)

                        # Backward pass + Optimize nếu trong phase train
                        if phase == 'train':
                            total_loss.backward()
                            optimizer.step()

                    # Cập nhật các thông số thống kê
                    running_loss += total_loss.item() * inputs.size(0)
                    running_loss_b += binary_loss.item() * inputs.size(0)
                    running_loss_i += instance_loss.item() * inputs.size(0)
                    binary_preds = torch.argmax(outputs['binary_seg_logits'], dim=1)
                    correct_binary += torch.sum(binary_preds == binarys).item()
                    false_positive_result += torch.sum((binary_preds == 1) & (binarys == 0)).item()
                    true_positive_result += torch.sum((binary_preds == 1) & (binarys == 1)).item()
                    total_pixels += binarys.numel()

                    # Cập nhật thanh tiến trình
                    pbar.update(1)

            # Tính toán thống kê cho mỗi epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            binary_accuracy = correct_binary / total_pixels

            # Precision, Recall, F1 calculations
            binary_total_false = total_pixels - correct_binary
            binary_precision = true_positive_result / (true_positive_result + false_positive_result) if (true_positive_result + false_positive_result) != 0 else 0
            binary_recall = true_positive_result / (true_positive_result + binary_total_false - false_positive_result) if (true_positive_result + binary_total_false - false_positive_result) != 0 else 0
            binary_f1_score = (2 * binary_precision * binary_recall) / (binary_precision + binary_recall) if (binary_precision + binary_recall) != 0 else 0

            print(f'{phase.capitalize()} Total Loss: {epoch_loss:.4f} Binary Loss: {binary_loss:.4f} Instance Loss: {instance_loss:.4f} Accuracy: {binary_accuracy:.4f} F1-Score: {binary_f1_score:.4f}')

            # Lưu mô hình tốt nhất dựa trên validation loss
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = model.state_dict()
                best_model_save_filename = os.path.join(save_path, 'best_model.pth')
                torch.save(best_model_wts, best_model_save_filename)
                print(f"Best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}")

    # Lưu checkpoint sau khi huấn luyện
    model_save_filename = os.path.join(save_path, 'checkpoint.pth')
    checkpoint = {
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, model_save_filename)
    print("Checkpoint is saved: {}".format(model_save_filename))

if __name__ == '__main__':
    train()
