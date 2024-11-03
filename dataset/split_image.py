import os
import shutil

input_dir = 'dataset/MOVI1818_image'
path_dir = "dataset_test/clips/data"

os.makedirs(path_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith('.jpg'):
        image_path = os.path.join(input_dir, file)
        image_path_new = os.path.join(path_dir, file)
        shutil.copy(image_path, image_path_new)
        print("Copying " + image_path_new)