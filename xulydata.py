import os
import shutil

def fix_paths(txt_file, prefix='./data/training_data_subset2/'):
    """
    Sửa các đường dẫn trong tệp txt để bao gồm chỉ một './' và thêm tiền tố.

    :param txt_file: Đường dẫn tới tệp txt (train.txt hoặc val.txt).
    :param prefix: Tiền tố cần thêm vào các đường dẫn.
    """
    if not os.path.exists(txt_file):
        print(f"Lỗi: Tệp {txt_file} không tồn tại.")
        return

    # Đọc nội dung tệp
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        paths = line.strip().split()
        new_paths = []
        for path in paths:
            # Loại bỏ tất cả './' ở đầu
            while path.startswith('./'):
                path = path[2:]
            
            # Thêm tiền tố
            new_path = os.path.join(prefix, path).replace('\\', '/')
            # Thêm './' vào đầu đường dẫn
            new_path = './' + new_path

            new_paths.append(new_path)
        
        # Tạo dòng mới
        new_line = ' '.join(new_paths) + '\n'
        new_lines.append(new_line)

    # Tạo bản sao lưu trước khi ghi đè
    backup_file = txt_file + '.backup'
    shutil.copyfile(txt_file, backup_file)
    print(f"Đã tạo bản sao lưu của tệp gốc tại {backup_file}")

    # Ghi lại tệp với các đường dẫn đã sửa
    with open(txt_file, 'w') as f:
        f.writelines(new_lines)

    print(f"Đã sửa các đường dẫn trong tệp {txt_file} thành công.")

# Đường dẫn tới thư mục gốc của dự án
project_root = r'D:\project2\lanenet-lane-detection-pytorch1'

# Đường dẫn tới các tệp txt cần sửa
train_txt = os.path.join(project_root, 'data', 'training_data_subset2', 'train.txt')
val_txt = os.path.join(project_root, 'data', 'training_data_subset2', 'val.txt')

# Gọi hàm để sửa các đường dẫn
fix_paths(train_txt)
fix_paths(val_txt)
