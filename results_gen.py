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
from utils import ensure_path, overlay_mask_on_image
from tqdm import tqdm
import os
import random
import shutil
from models.unet import ResUnetPlusPlus

def load_image_paths(root_path):
    image_paths = []
    image_extensions = ['*.jpg']
    for dirpath, _, _ in os.walk(root_path):
        for ext in image_extensions:
            files = glob.glob(os.path.join(dirpath, ext))
            for file_path in files:
                if 'bit' not in os.path.basename(file_path):
                    image_paths.append(file_path)

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
    for y in range(50, height, grid_spacing):
        for x in range(100, width, grid_spacing):
            if cv2.pointPolygonTest(max_contour, (x, y), False) > 0:
                points_inside_contour.append([x, y])

    for y in range(50, height, grid_spacing):
        for x in range(100, width, grid_spacing):
            if cv2.pointPolygonTest(max_contour, (x, y), False) < 0:
                points_outside_contour.append([x, y])

    input_point = np.array(points_inside_contour + points_outside_contour)
    input_label = [1]*len(points_inside_contour) + [0]*len(points_outside_contour)
    return input_point, input_label


def mask_filter(pre_bitmap):
    contours, _ = cv2.findContours(pre_bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(pre_bitmap)
    cv2.drawContours(mask, [max_contour], -1, (1), thickness=cv2.FILLED)
    
    return mask

def main():
    # Load sam model
    sam_checkpoint = "./src/sam_pretrained/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = torch.nn.DataParallel(sam).cuda()
    sam = sam.module

    predictor = SamPredictor(sam)

    Unet_checkpoint_Tool = "./save/Tool/Unet/best_model.pth"
    Unet_Tool = ResUnetPlusPlus(3)
    Unet_Tool.load_state_dict(torch.load(Unet_checkpoint_Tool))
    Unet_Tool = torch.nn.DataParallel(Unet_Tool).cuda()
    Unet_Tool = Unet_Tool.module

    Unet_checkpoint_PD = "./save/PD/Unet/best_model.pth"
    Unet_PD = ResUnetPlusPlus(3)
    Unet_PD.load_state_dict(torch.load(Unet_checkpoint_PD))
    Unet_PD = torch.nn.DataParallel(Unet_PD).cuda()
    Unet_PD = Unet_PD.module

    # Load image
    paths_Tool = load_image_paths('./src/Tool_Detection')
    paths_PD = load_image_paths('./src/PD_Impression_Detection')

    for img_path in tqdm(paths_Tool):
        gt_path = img_path.rsplit(".", 1)[0] + "_bitmap.jpg"
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
        # Overlay
        image_overlay = overlay_mask_on_image(image, pre_bitmap*255)
        # save image
        save_image(image, img_path.replace("src", "save/results"))
        save_image(pre_bitmap*255, gt_path.replace("src", "save/results").replace("_bitmap", "_sam"))
        save_image(image_overlay, img_path.replace("src", "save/results").replace(".jpg", "_sam_overlay.jpg"))

        # predict with Unet
        image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0)
        image_tensor = image_tensor.cuda()
        pre_bitmap = Unet_Tool(image_tensor)
        pre_bitmap = (pre_bitmap>0.5).squeeze().cpu().detach().numpy()
        pre_bitmap = cv2.resize(pre_bitmap.astype(np.uint8), (gt.shape[1],gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Overlay
        image_overlay = overlay_mask_on_image(image, pre_bitmap*255)
        # save image
        save_image(pre_bitmap*255, gt_path.replace("src", "save/results").replace("_bitmap", "_unet"))
        save_image(image_overlay, img_path.replace("src", "save/results").replace(".jpg", "_unet_overlay.jpg"))

    for img_path in tqdm(paths_PD):
        # predict with Unet
        image = np.array(Image.open(img_path).convert('RGB'))
        image_resized = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_NEAREST)
        image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0)
        image_tensor = image_tensor.cuda()
        pre_bitmap = Unet_PD(image_tensor)
        pre_bitmap = (pre_bitmap>0.5).squeeze().cpu().detach().numpy()
        pre_bitmap = cv2.resize(pre_bitmap.astype(np.uint8), (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Overlay
        image_overlay = overlay_mask_on_image(image, pre_bitmap*255)
        # save image
        save_image(image, img_path.replace("src", "save/results"))
        save_image(pre_bitmap*255, img_path.replace("src", "save/results").replace(".jpg", "_unet.jpg"))
        save_image(image_overlay, img_path.replace("src", "save/results").replace(".jpg", "_unet_overlay.jpg"))


if __name__ == '__main__':
    ensure_path('./save/results')
    main()
    
    