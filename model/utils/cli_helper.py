import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Lane Detection Models (LaneNet/SCNN/LaneATT)")

    # General arguments
    parser.add_argument("--dataset", required=True, help="Dataset path, which should contain train.txt and val.txt")
    parser.add_argument("--model_type", choices=['lanenet', 'scnn','laneatt'], help="Model type to train: lanenet or scnn", default='lanenet')
    parser.add_argument("--loss_type", help="Loss type", default='FocalLoss')
    parser.add_argument("--save", required=False, help="Directory to save model checkpoints and logs", default="./log")
    parser.add_argument("--epochs", required=False, type=int, help="Number of training epochs", default=25)
    parser.add_argument("--width", required=False, type=int, help="Resize width for input images", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height for input images", default=256)
    parser.add_argument("--bs", required=False, type=int, help="Batch size for training", default=4)
    parser.add_argument("--val", required=False, type=bool, help="Use validation during training", default=False)
    parser.add_argument("--lr", required=False, type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to the pretrained model file")

    # Arguments for LaneNet model
    parser.add_argument("--backbone", type=str, choices=['ENet', 'UNet', 'DeepLabv3+', 'LinkNet', 'BiSeNet', 'ERFNet'], 
                        default="ENet", help="Backbone type for LaneNet model (Only used if model_type is 'lanenet')")
    
    parser.add_argument("--laneatt_backbone", type=str, choices=['resnet50', 'resnet34', 'resnet18'],
                        default="resnet50", help="Backbone type for LaneATT model (Only used if model_type is 'laneatt')")
    # Output and testing related arguments
    parser.add_argument("--image", default="./output", help="Output folder for images (used during inference)")
    parser.add_argument("--net", help="Backbone network (not required for training)")
    parser.add_argument("--json", help="Post processing json file (for inference)")

    return parser.parse_args()
