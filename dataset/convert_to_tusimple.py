import os
import json

img_path = 'clips/data'
H_SAMPLES = list(range(0, 590, 10))

def convert_lane_to_tusimple_format(lanes, h_samples):
    """Chuyển đổi tọa độ làn đường sang định dạng TuSimple"""
    tusimple_lanes = []
    for lane in lanes:
        tusimple_lane = []
        for y in h_samples:
            # Tìm x tại y bằng cách nội suy hoặc gán -2 nếu không có điểm
            x = next((point[0] for point in lane if abs(point[1] - y) < 5), -2)
            tusimple_lane.append(int(x) if x != -2 else -2)
        tusimple_lanes.append(tusimple_lane)
    return tusimple_lanes

def convert_culane_to_custom_format(culane_labels_dir, output_json_path):
    data = []

    for label_file in os.listdir(culane_labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(culane_labels_dir, label_file)
            image_file = label_file.replace('.lines.txt', '.jpg')

            with open(label_path, 'r') as f:
                lines = f.readlines()

            lanes = []
            for line in lines:
                # Chia dòng thành các giá trị float và nhóm thành các cặp (x, y)
                points = list(map(float, line.strip().split()))
                lane = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                lanes.append(lane)

            tusimple_lane = convert_lane_to_tusimple_format(lanes, H_SAMPLES)

            # Tạo một dict cho mỗi tệp
            data.append({
                "raw_file": f"{img_path}/{image_file}",
                "h_samples": H_SAMPLES,
                "lanes": tusimple_lane  # Sử dụng lanes đã chuyển đổi
            })

    # Ghi dữ liệu vào tệp JSON
    with open(output_json_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f'Dữ liệu đã được chuyển đổi và lưu vào {output_json_path}')


# Sử dụng hàm
culane_labels_dir = 'dataset/MOVI1818_image'  # Đường dẫn tới thư mục chứa các tệp label
output_json_path = 'dataset_test/label_data_data.json'  # Đường dẫn tới tệp JSON đầu ra
convert_culane_to_custom_format(culane_labels_dir, output_json_path)
