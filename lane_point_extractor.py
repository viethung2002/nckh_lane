import os
import numpy as np
import cv2

def extract_lane_points(gt_instance_image, num_points=100):
    """
    Chuyển đổi từ ảnh gt_instance_image thành danh sách các điểm trên các làn đường.
    
    :param gt_instance_image: Ảnh nhãn phân vùng của các làn đường (ảnh grayscale)
    :param num_points: Số điểm cần lấy trên mỗi làn đường
    :return: Danh sách các điểm trên các làn đường dưới dạng (x, y, class_id)
    """
    gt_label_pts = []

    # Duyệt qua các giá trị khác nhau trong gt_instance_image
    unique_labels = np.unique(gt_instance_image)
    
    for label in unique_labels:
        if label == 0:
            # Bỏ qua giá trị 0 (giả sử là nền)
            continue
        
        # Trích xuất các pixel thuộc về làn đường hiện tại
        coords = np.column_stack(np.where(gt_instance_image == label))
        
        # Kiểm tra nếu số lượng điểm hiện tại ít hơn num_points
        if len(coords) <= num_points:
            sampled_coords = coords
        else:
            # Tính toán khoảng cách giữa các điểm sao cho có đúng num_points điểm
            indices = np.linspace(0, len(coords) - 1, num_points, dtype=int)
            sampled_coords = coords[indices]
        
        # Thêm vào gt_label_pts
        for coord in sampled_coords:
            x, y = coord[1], coord[0]  # Chuyển (row, col) thành (x, y)
            gt_label_pts.append([x, y, label])

    return gt_label_pts

def generate_gt_label_pts(data_dir, output_file, num_points=100):
    """
    Hàm chính để tạo file chứa tọa độ gt_label_pts từ các ảnh gt_instance_image.

    :param data_dir: Đường dẫn tới thư mục chứa các ảnh gt_instance_image.
    :param output_file: Đường dẫn tới file đầu ra chứa gt_label_pts.
    :param num_points: Số điểm cần lấy trên mỗi làn đường.
    """
    with open(output_file, 'w') as f:
        for filename in os.listdir(data_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                file_path = os.path.join(data_dir, filename)
                gt_instance_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Kiểm tra nếu ảnh đã được tải thành công
                if gt_instance_image is None:
                    print(f"Không thể tải ảnh: {file_path}")
                    continue

                # Trích xuất các điểm từ ảnh với số lượng num_points
                gt_label_pts = extract_lane_points(gt_instance_image, num_points)

                # Lưu kết quả vào file
                for pt in gt_label_pts:
                    x, y, label = pt
                    f.write(f"{filename},{x},{y},{label}\n")

                print(f"Đã xử lý ảnh: {filename}")

if __name__ == '__main__':
    # Đường dẫn tới thư mục chứa ảnh gt_instance_image
    data_dir = input("Nhập đường dẫn tới thư mục chứa ảnh gt_instance_image: ")
    
    # Hiển thị đường dẫn để kiểm tra
    print(f"Đường dẫn bạn đã nhập là: {data_dir}")

    # Đường dẫn tới file đầu ra chứa gt_label_pts
    output_file = "gt_label_pts.txt"

    # Số điểm cần lấy trên mỗi làn đường
    num_points = 100

    # Tạo file chứa gt_label_pts với số điểm cố định
    generate_gt_label_pts(data_dir, output_file, num_points)
    print(f"Hoàn tất! Kết quả được lưu tại: {output_file}")
