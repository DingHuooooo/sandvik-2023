import os
import json
import cv2
import numpy as np
from PIL import Image
from skimage.draw import polygon
import random
from utils import ensure_path

def save_image(img, path):
    if img.dtype != np.uint8:
        print("Error: Image dtype should be uint8.")
        return
    if img.min() < 0 or img.max() > 255:
        print("Error: Pixel values should be in [0, 255].")
        return
    if len(img.shape) == 3:
        if img.shape[0] == 3 or img.shape[0] == 4:  
            img = img.transpose(1, 2, 0)

    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(path, img_bgr)
    if not success:
        print("Error: Image failed to save.")

def json_to_mask(json_path, img_shape):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        mask = np.zeros(img_shape, dtype=np.uint8)
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                rr, cc = polygon(points[:, 1], points[:, 0])
                mask[rr, cc] = 1
        
        return Image.fromarray((mask * 255).astype(np.uint8))
    else:
        empty_mask = np.zeros(img_shape, dtype=np.uint8)
        return Image.fromarray(empty_mask)

def rotate_and_save(img, mask, base_path, suffix):
    # Resize to 1024x1024
    resized_img = cv2.resize(img, (1024, 1024))
    resized_mask = cv2.resize(mask, (1024, 1024))

    for angle in [0, 90, 180, 270]:
        rotated_img = np.array(Image.fromarray(resized_img).rotate(angle))
        rotated_mask = np.array(Image.fromarray(resized_mask).rotate(angle))
        save_image(rotated_img, f"{base_path}_rotate{angle}.jpg")
        save_image(rotated_mask, f"{base_path.replace(suffix, f'{suffix}_gt')}_rotate{angle}_bitmap.jpg")

def process_directory(root_dir):
    for subdir, _, files in os.walk(root_dir):
        if 'Flank' in subdir:
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(subdir, file)
                    json_path = img_path.replace('.jpg', '.json')
                    
                    img = np.array(Image.open(img_path))
                    img_shape = (img.shape[0], img.shape[1])
                    
                    mask = np.array(json_to_mask(json_path, img_shape))
                    
                    if random.random() < 0.8:  # 80% chance for train
                        dest_path = './save/PD/train/' + os.path.basename(img_path).replace('.jpg', '')
                        rotate_and_save(img, mask, dest_path, 'train')
                    else:  # 20% chance for val
                        dest_path = './save/PD/val/' + os.path.basename(img_path).replace('.jpg', '')
                        rotate_and_save(img, mask, dest_path, 'val')

root_dir = './src/PD_Impression_Detection'
ensure_path('./save/PD/')
process_directory(root_dir)
