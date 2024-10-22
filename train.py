import time
import os
import sys

import torch
from model.lanenet.train_lanenet import train_model as train_lanenet_model  # Import hàm huấn luyện cho LaneNet
from model.csnn.train_csnn import train_model as train_csnn_model  # Import hàm huấn luyện cho CSNN
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from model.csnn.LaneDetectionCSNN import LaneDetectionWithCSNN  # Import mô hình CSNN
from torch.utils.data import DataLoader
from torchvision import transforms
from model.utils.cli_helper import parse_args
import pandas as pd

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
        model = LaneNet(arch=args.backbone)  # Khởi tạo LaneNet với backbone được chỉ định
        train_fn = train_lanenet_model  # Sử dụng hàm huấn luyện của LaneNet
    elif args.model_type.lower() == 'csnn':
        model = LaneDetectionWithCSNN()  # Khởi tạo CSNN
        train_fn = train_csnn_model  # Sử dụng hàm huấn luyện của CSNN
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
    model, log = train_fn(
        model,
        optimizer,
        scheduler=None,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=DEVICE,
        loss_type=args.loss_type,
        num_epochs=args.epochs,
        start_epoch=start_epoch
    )

    # Kiểm tra và lưu mô hình tốt nhất
    for epoch, val_loss in zip(log['epoch'], log['val_loss']):
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            best_model_save_filename = os.path.join(save_path, 'best_model.pth')
            torch.save(best_model_wts, best_model_save_filename)
            print(f"Best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}")

    # Lưu training log
    df = pd.DataFrame({
        'epoch': log['epoch'],
        'training_loss': log['training_loss'],
        'val_loss': log['val_loss']
    })

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch', 'training_loss', 'val_loss'],
              header=True, index=False, encoding='utf-8')
    print("Training log is saved: {}".format(train_log_save_filename))

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
