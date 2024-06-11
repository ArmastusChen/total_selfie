
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 
import numpy as np 
import PIL
from PIL import Image
from utils import *
import glob
import argparse

from utils import run_seg, get_seg_by_name, get_crop_region
from misc import  get_folder_name



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--mask_type', type=str, default='rectangle')
parser.add_argument('--mask_expand_ratio', type=float, default=1.0)
parser.add_argument('--guidance_scale', type=float, default=5.0)
parser.add_argument('--total', type=int, default=200)
parser.add_argument('--skip_shoes', action='store_true')
parser.add_argument('--skip_seg', action='store_true')
parser.add_argument('--use_y', action='store_true')
parser.add_argument('--model_id', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None )


args = parser.parse_args()




# set the parts porportions
parts = ['top', 'bottom', 'face', 'top', 'bottom', 'face','shoes']

np.random.seed(0)

name = args.name
mask_type = args.mask_type
mask_expand_ratio = args.mask_expand_ratio
guidance_scale = args.guidance_scale
skip_seg = args.skip_seg
use_y = args.use_y
output_dir = args.output_dir
total = args.total
model_id = args.model_id

model_name = model_id.split('/')[-1]

folder_name = get_folder_name(name, model_name, mask_type, mask_expand_ratio, guidance_scale)

full_body_dir = f'{output_dir}/results/no_cond/{folder_name}'
cache_dir = f'cache/{folder_name}/seg'
out_dir = f'./{output_dir}/fine_tune_pbe_data/{folder_name}'
input_dir = f'{out_dir}/input'
scene_dir = f'{out_dir}/scene'

out_gt_dir = f'{out_dir}/gt'
out_mask_dir = f'{out_dir}/mask'
out_loss_mask_dir = f'{out_dir}/loss_mask'

os.makedirs(out_gt_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)
os.makedirs(out_loss_mask_dir, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)
os.makedirs(scene_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# only get the full body images with name can be converted to int
full_body_paths = glob.glob(full_body_dir + '/*')
full_body_paths = [full_body_path for full_body_path in full_body_paths if full_body_path.split('/')[-1][:-4].isdigit()]


if not skip_seg:
    # detect the seg of generated uncond full body images
    for full_body_path in full_body_paths:
        seg_name = full_body_path.split('/')[-1][:-4] 
        full_body = Image.open(full_body_path)
        full_body = np.array(full_body)

        seg, _ = run_seg(full_body)

        seg = Image.fromarray(seg)
        seg.save(f'{cache_dir}/{seg_name}.png')



#  copy input selfies 
image_paths = []
for part in parts:
    if part == 'face':
        image_paths.append(f'./data/{name}/processed/face_corrected_a.png')
    else:
        image_paths.append(f'./data/{name}/processed/{part}.png')

    cmd = f'cp -r {image_paths[-1]} {input_dir}/{part}.png'
    os.system(cmd)

    
# we crop the selfies before placing them on the scene. We don't crop the shoes because they are already cropped. 
crop_images = {}
for image_path in image_paths:

    if 'shoes' in image_path:
        image = Image.open(image_path)
        crop_image = np.array(image)

        key = image_path.split('/')[-1][:-4]

    
    else:
        image = Image.open(image_path)
        image = np.array(image)
        key = image_path.split('/')[-1][:-4]

        if 'face' in key:
            key = 'face'

        seg, _ = run_seg(np.array(image))

        seg_mask = get_seg_by_name(seg, key,)
        crop_region = get_crop_region(seg_mask)
        x1, y1, x2, y2 = crop_region

        crop_image = image[y1:y2, x1:x2]

    crop_images[key] = crop_image




'''
Start to generate augmented data by placing the cropped selfies on the masked scene
'''

seg_paths = glob.glob(cache_dir + '/*')


# get the scene
scene_path = f'./data/{name}/processed/scene.png'
scene = Image.open(scene_path)
scene.save(f'{scene_dir}/scene.png')
scene = np.array(scene)


for i in range(total):
    # random select a part from crop_images
    part = parts[i % len(parts)]
    
    crop_image = crop_images[part]


    
    # resize crop_image, matching image resolution of x-axis, keep aspect ratio
    while True:
        # random select a prior_seg
        seg_path = np.random.choice(seg_paths)
        seg = Image.open(seg_path)
        prior_seg = np.array(seg)

        # get the mask 
        prior_mask = Image.open(f'{full_body_dir}/mask.png').resize((512,512), Image.NEAREST).convert('RGB')
        prior_mask = np.array(prior_mask)

        masked_scene = scene * (1 - prior_mask / 255)
        masked_scene = np.uint8(masked_scene)

        # get the mask of part from prior_seg
        if part == 'face':
            prior_part_mask = get_seg_by_name(prior_seg, 'face-hair')
        else:
            prior_part_mask = get_seg_by_name(prior_seg, part)

        # get the center of prior_part_mask using mean 
        cy, cx = np.where(prior_part_mask == 1)
        cy, cx = np.mean(cy), np.mean(cx)
        cy, cx = int(cy), int(cx)
        

        # get bbox of prior_part_mask
        crop_region = get_crop_region(prior_part_mask)
        x1, y1, x2, y2 = crop_region 

        crop_region_whole = get_crop_region(prior_mask[..., 0])
        x1_w, y1_w, x2_w, y2_w = crop_region_whole 



        # sample centriod of the placement, and scale of the crop_image 
        if part in ['face']:
            x_diff = x2 - x1 + (x2 - x1) * np.random.randint(3, 10) / 100
            cx_cur = cx - cx * np.random.randint(-5, 5) / 100
            cy_cur = cy - cy * np.random.randint(-5, 5) / 100
        elif part in ['top', 'bottom']:
            if not use_y:
                x_diff = x2 - x1 + (x2 - x1) * np.random.randint(-15, 15) / 100
            else:
                x_diff = y2 - y1 + (y2 - y1) * np.random.randint(-15, 15) / 100

            cx_cur = cx - cx * np.random.randint(-15, 15) / 100
            cy_cur = cy - cy * np.random.randint(-15, 15) / 100

        elif part in ['shoes']:
            x_diff = x2 - x1 + (x2 - x1) * np.random.randint(-3, 3) / 100

            cx_cur = cx - cx * np.random.randint(-3, 3) / 100
            cy_cur = cy - cy * np.random.randint(-3, 3) / 100

        x_diff = int(x_diff)

        crop_image_cur = crop_image.copy()

        if part in ['shoes']:
            # keep aspect ratio 
            ratio = (y2 - y1) / (x2 - x1)

            crop_image_cur = cv2.resize(crop_image_cur, (x_diff, int(x_diff * ratio)))
        else:
            if part in ['top', 'bottom'] and use_y:
                crop_image_cur = cv2.resize(crop_image_cur, (x_diff * crop_image.shape[1] // crop_image.shape[0], x_diff), interpolation=cv2.INTER_NEAREST)
            else:
                crop_image_cur = cv2.resize(crop_image_cur, (x_diff, x_diff * crop_image.shape[0] // crop_image.shape[1]), interpolation=cv2.INTER_NEAREST)



        cx_cur, cy_cur = int(cx_cur), int(cy_cur)


        # paste crop_image_cur to masked_scene based on cx and cy
        x_min, x_max = cx_cur - crop_image_cur.shape[1] // 2, cx_cur + crop_image_cur.shape[1] // 2
        y_min, y_max = cy_cur - crop_image_cur.shape[0] // 2, cy_cur + crop_image_cur.shape[0] // 2

        # if exceed the mask, try another scale
        if x_min < x1_w or x_max > x2_w or y_min < y1_w or y_max > y2_w:
            print(f'exceed the mask, try another scale for {part}')
            continue

        break


    
  
    # make sure crop_image_cur and masked_scene have the same size
    crop_image_cur = crop_image_cur[:y_max - y_min, :x_max - x_min]

    
    masked_scene[y_min:y_max, x_min:x_max] = crop_image_cur


    # save 
    masked_scene = Image.fromarray(masked_scene)
    masked_scene.save(f'{out_gt_dir}/{part}_{i}.png')

    prior_mask = Image.fromarray(prior_mask)
    prior_mask.save(f'{out_mask_dir}/{part}_{i}.png')

    loss_mask = 255 - np.array(prior_mask)
    loss_mask[y_min:y_max, x_min:x_max] = 255
    # loss_mask = loss_mask * prior_mask
    loss_mask = np.uint8(loss_mask)

    loss_mask = Image.fromarray(loss_mask)
    loss_mask.save(f'{out_loss_mask_dir}/{part}_{i}.png')

    

    
    


    







    
    



