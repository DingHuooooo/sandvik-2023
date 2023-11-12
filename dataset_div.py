import os
import random
import shutil
import numpy as np
from PIL import Image

def load_image_paths(root_path):
    image_paths = []
    image_extensions = ['.jpg', '.png']  # Note the change here to just the extension
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(tuple(image_extensions)) and 'label' not in file:
                image_paths.append(os.path.join(dirpath, file))

    return image_paths

def rotate_and_save_image(image_path, rotation, save_dir):
    image = Image.open(image_path)
    rotated_image = image.rotate(rotation)
    base_name = os.path.basename(image_path)
    new_name = f"{os.path.splitext(base_name)[0]}_{rotation}.png"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rotated_image.save(os.path.join(save_dir, new_name))

def dataset_div(src_image_dir):
    
    image_paths = load_image_paths(src_image_dir)
    num_val_samples = int(len(image_paths) * 0.2)
    val_files = random.sample(image_paths, num_val_samples)

    for file_path in image_paths:
        wear_path = file_path.replace('.jpg', '_label.png')

        if file_path in val_files:
            # Rotate and save validation images
            for angle in [0]:
                rotate_and_save_image(wear_path, angle, './src/dataset/PDImpression/val_gt')
                rotate_and_save_image(file_path, angle, './src/dataset/PDImpression/val')
        else:
            # Rotate and save training images
            for angle in [0, 90, 180, 270]:
                rotate_and_save_image(wear_path, angle, './src/dataset/PDImpression/train_gt')
                rotate_and_save_image(file_path, angle, './src/dataset/PDImpression/train')
        

if __name__ == '__main__':
    dataset_div('/home/mr634151/Sandvik_2023/src/Wear_detection/PDImpression')