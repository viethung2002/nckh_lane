import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from model.lanenet.LaneNet import LaneNet
from model.lanenet.Hnet import HNet
from model.lanenet.lanenet_postprocess import LaneNetPostProcessor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Path to image, video, or folder containing images", required=True)
    parser.add_argument("--model_type", choices=['lanenet', 'scnn', 'laneatt'], help="Model type for testing", default='lanenet')
    parser.add_argument("--backbone", help="Backbone type for models (LaneNet, LaneATT, etc.)", default='resnet50')
    parser.add_argument("--model", help="Path to trained model", required=True)
    parser.add_argument("--hnet_model", help="Path to trained HNet model", required=True)
    parser.add_argument("--width", type=int, help="Width for resizing", default=512)
    parser.add_argument("--height", type=int, help="Height for resizing", default=256)
    parser.add_argument("--save", help="Folder to save results", default="./test_output")
    return parser.parse_args()

def load_test_data(img, transform):
    img = transform(img)
    return img

def process_frame(frame, model, postprocessor, data_transform, resize_width, resize_height):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dummy_input = load_test_data(img, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    
    with torch.no_grad():
        outputs = model(dummy_input)

    # Post-process the model outputs
    mask, overlay = postprocessor.postprocess(
        binary_seg_pred=outputs['binary_seg_pred'],
        instance_seg_logits=outputs['instance_seg_logits'],
        source_image=np.array(img)
    )

    if overlay is not None:
        return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    else:
        return frame

def process_video(video_path, model, postprocessor, data_transform, resize_width, resize_height, save_dir):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(save_dir, 'processed_video.avi'), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (resize_width, resize_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_width, resize_height))
        processed_frame = process_frame(frame, model, postprocessor, data_transform, resize_width, resize_height)
        out.write(processed_frame)

    cap.release()
    out.release()

def process_image(image_path, model, postprocessor, data_transform, resize_width, resize_height, save_dir):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((resize_width, resize_height))
    processed_img = process_frame(np.array(img), model, postprocessor, data_transform, resize_width, resize_height)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, processed_img)

def test():
    args = parse_args()
    os.makedirs(args.save, exist_ok=True)

    resize_height = args.height
    resize_width = args.width
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model_type = args.model_type.lower()

    if model_type == 'lanenet':
        model = LaneNet(arch=args.backbone)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Error loading model state dict: {e}")

    model.eval()
    model.to(DEVICE)

    postprocessor = LaneNetPostProcessor(hnet_model_path=args.hnet_model, device=DEVICE)

    if args.img.endswith(('.mp4', '.avi', '.mov')):
        process_video(args.img, model, postprocessor, data_transform, resize_width, resize_height, args.save)
    elif os.path.isdir(args.img):
        img_files = [os.path.join(args.img, f) for f in os.listdir(args.img) if f.endswith(('jpg', 'png', 'jpeg'))]
        for img_file in img_files:
            process_image(img_file, model, postprocessor, data_transform, resize_width, resize_height, args.save)
    else:
        process_image(args.img, model, postprocessor, data_transform, resize_width, resize_height, args.save)

if __name__ == "__main__":
    test()
