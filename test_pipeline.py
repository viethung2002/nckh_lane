import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from model.lanenet.LaneNet import LaneNet
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from model.lanenet.Hnet import HNet
from model.lanenet.lanenet_postprocess import LaneNetPostProcessor


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Đường dẫn tới ảnh", required=True)
    parser.add_argument("--model_type", choices=['lanenet', 'scnn', 'laneatt'], help="Loại mô hình để kiểm tra", default='lanenet')
    parser.add_argument("--backbone", help="Loại backbone cho các mô hình (LaneNet, LaneATT, v.v.)", default='resnet50')
    parser.add_argument("--model", help="Đường dẫn tới mô hình đã huấn luyện", required=True)
    parser.add_argument("--hnet_model", help="Đường dẫn tới mô hình HNet đã huấn luyện", required=True)
    parser.add_argument("--width", type=int, help="Chiều rộng khi resize", default=512)
    parser.add_argument("--height", type=int, help="Chiều cao khi resize", default=256)
    parser.add_argument("--save", help="Thư mục để lưu kết quả", default="./test_output")
    return parser.parse_args()

def load_test_data(img_path, transform):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img

def test():
    args = parse_args()
    
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(args.save, exist_ok=True)
    
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    # Định nghĩa các phép biến đổi cho ảnh
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model_type = args.model_type.lower()

    # Khởi tạo mô hình LaneNet
    if model_type == 'lanenet':
        model = LaneNet(arch=args.backbone)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Tải state dictionary của mô hình LaneNet
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(f"Lỗi khi tải state dict của mô hình: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")

    model.eval()
    model.to(DEVICE)

    # Khởi tạo post-processor với đường dẫn tới mô hình HNet
    postprocessor = LaneNetPostProcessor(hnet_model_path=args.hnet_model, device=DEVICE)

    # Tải và tiền xử lý ảnh thử nghiệm
    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    
    # Thực hiện forwarding qua mô hình LaneNet
    with torch.no_grad():
        outputs = model(dummy_input)

    # Chuẩn bị ảnh gốc để trực quan hóa
    input_img = Image.open(img_path).convert('RGB')
    input_img = input_img.resize((resize_width, resize_height))
    input_img = np.array(input_img)

    # Hậu xử lý kết quả đầu ra của mô hình
    mask, overlay = postprocessor.postprocess(
        binary_seg_pred=outputs['binary_seg_pred'],
        instance_seg_logits=outputs['instance_seg_logits'],
        source_image=input_img
    )

    # Lưu các ảnh kết quả
    cv2.imwrite(os.path.join(args.save, 'input.jpg'), cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
    if mask is not None:
        cv2.imwrite(os.path.join(args.save, 'mask_output.jpg'), mask)
    if overlay is not None:
        cv2.imwrite(os.path.join(args.save, 'overlay_output.jpg'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    test()
