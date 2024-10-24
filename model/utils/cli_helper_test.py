import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Img path")
    parser.add_argument("--model_type", choices=['lanenet', 'scnn', 'laneatt'], help="Model type to test: lanenet, scnn, or laneatt", default='lanenet')
    parser.add_argument("--backbone", type=str, choices=['ENet', 'UNet', 'DeepLabv3+', 'LinkNet', 'BiSeNet', 'ERFNet'], 
                        default="ENet", help="Backbone type for LaneNet model (Only used if model_type is 'lanenet')")
    
    parser.add_argument("--laneatt_backbone", type=str, choices=['resnet50', 'resnet34', 'resnet18'],
                        default="resnet50", help="Backbone type for LaneATT model (Only used if model_type is 'laneatt')")
    parser.add_argument("--model", help="Model path", default='./log/best_model.pth')
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--save", help="Directory to save output", default="./test_output")
    return parser.parse_args()

    
    
