import os
import numpy as np
import cv2

def draw_lane_points_on_image(image_dir, gt_label_pts_file, output_dir):
    """
    Vẽ các điểm làn đường lên ảnh gốc từ dữ liệu gt_label_pts.
    
    :param image_dir: Đường dẫn tới thư mục chứa ảnh gốc.
    :param gt_label_pts_file: Đường dẫn tới file gt_label_pts.
    :param output_dir: Đường dẫn tới thư mục lưu trữ các ảnh đầu ra.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Đọc file gt_label_pts
    with open(gt_label_pts_file, 'r') as file:
        lines = file.readlines()

    # Sử dụng một từ điển để nhóm các điểm theo ảnh gốc
    images_points = {}

    # Duyệt qua từng dòng để thu thập các điểm
    for line in lines:
        # Mỗi dòng có định dạng: image_name,x,y,class_id
        line_parts = line.strip().split(',')
        image_name = line_parts[0]
        x = int(line_parts[1])
        y = int(line_parts[2])
        class_id = int(line_parts[3])  # class_id có thể được sử dụng để phân biệt các làn đường khác nhau

        if image_name not in images_points:
            images_points[image_name] = []

        # Lưu điểm vào từ điển theo tên ảnh
        images_points[image_name].append((x, y, class_id))

    # Duyệt qua từng ảnh và vẽ tất cả các điểm đã thu thập
    for image_name, points in images_points.items():
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        # Kiểm tra xem ảnh gốc có tồn tại không
        if not os.path.exists(image_path):
            print(f"Ảnh không tồn tại: {image_path}")
            continue
        
        # Đọc ảnh gốc
        image = cv2.imread(image_path)
        
        # Vẽ tất cả các điểm lên ảnh
        for (x, y, class_id) in points:
            color = (0, 0, 255)  # Màu đỏ cho các điểm (có thể thay đổi theo `class_id`)
            thickness = 2
            cv2.circle(image, (x, y), 2, color, thickness)
        
        # Lưu ảnh sau khi đã vẽ
        cv2.imwrite(output_path, image)

        print(f"Đã vẽ các điểm lên ảnh và lưu tại: {output_path}")

if __name__ == '__main__':
    # Đường dẫn tới thư mục chứa ảnh gốc
    image_dir = input("Nhập đường dẫn tới thư mục chứa ảnh gốc: ")
    
    # Đường dẫn tới file gt_label_pts
    gt_label_pts_file = input("Nhập đường dẫn tới file gt_label_pts: ")
    
    # Đường dẫn tới thư mục lưu trữ các ảnh đầu ra
    output_dir = input("Nhập đường dẫn tới thư mục lưu trữ ảnh đầu ra: ")
    
    # Vẽ các điểm lên ảnh gốc
    draw_lane_points_on_image(image_dir, gt_label_pts_file, output_dir)
