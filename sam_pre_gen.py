import torch
import numpy as np
import torch.nn as nn
from models import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import glob
from utils import ensure_path
from tqdm import tqdm
import os
import random
import shutil

def load_image_paths(root_path):
    image_paths = []
    image_extensions = ['.jpg', '.png']  # Note the change here to just the extension
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(tuple(image_extensions)) and 'bit' not in file:
                image_paths.append(os.path.join(dirpath, file))

    return image_paths

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

def points_gen(pre_bitmap):
    contours, _ = cv2.findContours(pre_bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(pre_bitmap)
    cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)

    grid_spacing = 50
    height, width = mask.shape
    points_inside_contour = []
    points_outside_contour = []
    for y in range(10, height, int(grid_spacing)):
        for x in range(100, width, 4*grid_spacing):
            if cv2.pointPolygonTest(max_contour, (x, y), False) > 0:
                points_inside_contour.append([x, y])

    for y in range(10, height, grid_spacing):
        for x in range(50, width, 4*grid_spacing):
            if cv2.pointPolygonTest(max_contour, (x, y), False) < 0:
                points_outside_contour.append([x, y])

    input_point = np.array(points_inside_contour + points_outside_contour)
    input_label = [1]*len(points_inside_contour) + [0]*len(points_outside_contour)

    # Calculate the resizing factors
    resize_factor_x = 1024 / width
    resize_factor_y = 1024 / height
    input_points_resized = [(int(x * resize_factor_x), int(y * resize_factor_y)) for x, y in input_point]
    input_points_resized = np.array(input_points_resized)

    return input_point, input_label


def mask_filter(pre_bitmap):
    contours, _ = cv2.findContours(pre_bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(pre_bitmap)
    cv2.drawContours(mask, [max_contour], -1, (1), thickness=cv2.FILLED)
    
    return mask

def dataset_div():
    src_image_dir = './save/Tool/train/'
    src_gt_dir = './save/Tool/train_gt_sam/'

    dst_image_dir = './save/Tool/val'
    dst_gt_dir = './save/Tool/val_gt_sam'
    if not os.path.exists(dst_image_dir):
        os.mkdir(dst_image_dir)
        os.mkdir(dst_gt_dir)

    image_files = os.listdir(src_image_dir)

    num_val_samples = int(len(image_files) * 0.2)
    val_files = random.sample(image_files, num_val_samples)

    for file_name in val_files:
        shutil.move(os.path.join(src_image_dir, file_name), os.path.join(dst_image_dir, file_name))
        shutil.move(os.path.join(src_gt_dir, file_name.rsplit(".", 1)[0] + "_bitmap.jpg"), os.path.join(dst_gt_dir, file_name.rsplit(".", 1)[0] + "_bitmap.jpg"))

def main():

    # Load sam model
    sam_checkpoint = "./src/sam_pretrained/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = torch.nn.DataParallel(sam).cuda()
    sam = sam.module
    predictor = SamPredictor(sam)
    # Load image
    paths = load_image_paths('./src/Tool_detection')
    for img_path in tqdm(paths):
        gt_path = img_path.rsplit(".", 1)[0] + "_bitmap.png"
        image = np.array(Image.open(img_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).convert('L'))

        image_resized = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_NEAREST)
        gt_resized = cv2.resize(gt, (1024,1024), interpolation=cv2.INTER_NEAREST)
        outputs_lowres = nn.MaxPool2d(4)(torch.tensor(gt_resized).unsqueeze(0).float())
        # predict with sam
        predictor.set_image(image_resized)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            mask_input=outputs_lowres,
            multimask_output=True,
        )
        pre_bitmap = cv2.resize(masks[0].astype(np.uint8), (gt.shape[1],gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        input_point, input_label = points_gen(pre_bitmap)
        pre_bitmap_resized = cv2.resize(pre_bitmap*255, (1024,1024), interpolation=cv2.INTER_NEAREST)
        outputs_lowres = nn.MaxPool2d(4)(torch.tensor(pre_bitmap_resized).unsqueeze(0).float())
        # predict with sam, adding points
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=outputs_lowres,
            multimask_output=True,
        )
        predictor.reset_image()
        pre_bitmap = cv2.resize(masks[0].astype(np.uint8), (gt.shape[1],gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        pre_bitmap = mask_filter(pre_bitmap)
        # save image
        save_image(image, img_path.replace('src', 'save'))
        save_image(pre_bitmap*255, gt_path.replace('src', 'save').replace('_bitmap', '_sampre_bitmap'))
        # Overlay mask
        pre_overlay = cv2.addWeighted(image, 0.8, np.stack((pre_bitmap*255,)*3, axis=-1), 0.2, 0)
        save_image(pre_overlay, gt_path.replace('src', 'save').replace('_bitmap', '_overlay'))


if __name__ == '__main__':
    ensure_path('./save/Tool_detection')
    main()
    # dataset_div()
    