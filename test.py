import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Img path")
    parser.add_argument("--model_type", choices=['lanenet', 'scnn', 'laneatt'], help="Model type to test: lanenet, scnn, or laneatt", default='lanenet')
    parser.add_argument("--backbone", help="Backbone type for models (LaneNet, LaneATT, etc.)", default='resnet50')
    parser.add_argument("--model", help="Model path", required=True)
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--save", help="Directory to save output", default="./test_output")
    return parser.parse_args()

import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from model.laneatt.LaneATT import LaneATT
from model.scnn.LaneDetectionSCNN import LaneDetectionWithSCNN
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img

def test():
    args = parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model_type = args.model_type.lower()

    # Khởi tạo mô hình dựa trên `model_type`
    if model_type == 'lanenet':
        model = LaneNet(arch=args.backbone) if model_type == 'lanenet' else None
    elif model_type == 'scnn':
        model = LaneDetectionWithSCNN()
    elif model_type == 'laneatt':
        model = LaneATT(backbone=args.backbone)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load mô hình và đặt chế độ đánh giá
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model state dict: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.eval()
    model.to(DEVICE)

    # Load ảnh thử nghiệm
    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    # Xử lý kết quả đầu ra
    input_img = Image.open(img_path)
    input_img = input_img.resize((resize_width, resize_height))
    input_img = np.array(input_img)

    instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

    # Lưu kết quả
    cv2.imwrite(os.path.join(args.save, 'input.jpg'), input_img)
    if len(instance_pred.shape) == 3:
        cv2.imwrite(os.path.join(args.save, 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
    else:
        cv2.imwrite(os.path.join(args.save, 'instance_output.jpg'), instance_pred)
    cv2.imwrite(os.path.join(args.save, 'binary_output.jpg'), binary_pred)

if __name__ == "__main__":
    test()
