import json
import random


def sample_json_data(input_json_path, output_json_path, sample_size=20):
    with open(input_json_path, 'r') as json_file:
        data = json_file.readlines()
    with open(output_json_path, 'w') as f:
        for line in data:
            sample_size -= 1
            f.write(line)
            if sample_size == 0:
                break
# Sử dụng hàm
input_json_path = 'dataset_test/label_data_data.json'  # Đường dẫn tới tệp JSON đầu vào
output_json_path = 'dataset_test/test_label.json'  # Đường dẫn tới tệp JSON đầu ra
sample_json_data(input_json_path, output_json_path)
