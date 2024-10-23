import time
import os
import sys

import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torchvision import transforms
from model.utils.cli_helper_eval import parse_args
from model.eval_function import Eval_Score
from model.utils.train_utils import calculate_map  # Import the calculate_map function

import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluation():
    args = parse_args()
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    dataset_file = os.path.join(args.dataset, 'test.txt')
    Eval_Dataset = TusimpleSet(dataset_file, transform=data_transform, target_transform=target_transforms)
    eval_dataloader = DataLoader(Eval_Dataset, batch_size=1, shuffle=True)

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    iou, dice, map_score = 0, 0, 0  # Initialize mAP

    with torch.no_grad():
        for x, target, _ in eval_dataloader:
            x = x.to(DEVICE)
            y = model(x)

            y_pred = torch.squeeze(y['binary_seg_pred'].to('cpu')).numpy()
            y_true = torch.squeeze(target).numpy()

            # Calculate Dice and IoU using existing Eval_Score class
            Score = Eval_Score(y_pred, y_true)
            dice += Score.Dice()
            iou += Score.IoU()

            # Calculate mAP using calculate_map function
            y_prob = torch.sigmoid(y['binary_seg_pred'])  # Apply sigmoid to get probability
            mean_ap = calculate_map(y_prob.cpu(), target)
            map_score += mean_ap

    num_samples = len(eval_dataloader.dataset)
    print('Final_IoU: {:.4f}'.format(iou / num_samples))
    print('Final_F1: {:.4f}'.format(dice / num_samples))
    print('Final_mAP: {:.4f}'.format(map_score / num_samples))  # Print the final mAP

if __name__ == "__main__":
    evaluation()
