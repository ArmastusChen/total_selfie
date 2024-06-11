
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
parser.add_argument('--augment_num', type=int, default=50)
parser.add_argument('--output_dir', type=str, default=None )


args = parser.parse_args()

parts = ['face_corrected_a','shoes']


name = args.name
mask_type = args.mask_type
mask_expand_ratio = args.mask_expand_ratio
guidance_scale = args.guidance_scale
output_dir = args.output_dir
augment_num = args.augment_num



np.random.seed(0)

our_dir = f'./{output_dir}/fine_tune_db_data/{name}'


for part in parts:
    if 'face' in part:
        min_resize_res = 350 
        max_resize_res = 450

    if 'shoes' in part:
        min_resize_res = 400 
        max_resize_res = 500


    img_path = f'./data/{name}/processed/{part}.png'
    img = Image.open(img_path).convert('RGB').resize((512,512))

    our_dir_cur = f'{our_dir}/{part}'
    os.makedirs(our_dir_cur, exist_ok=True)

    img.save(f'{our_dir_cur}/input.png')



    for augment_ind in range(augment_num):

        # black image
        augmented_img = Image.new('RGB', (512,512), (0, 0, 0))


        # resize img 


        resize_res = np.random.randint(min_resize_res, max_resize_res)
        resize_img = img.resize((resize_res, resize_res))


        # randomly place it in the augmented image

        x = np.random.randint(0, 512 - resize_res)
        y = np.random.randint(0, 512 - resize_res)


        augmented_img.paste(resize_img, (x, y))

        augmented_img.save(f'{our_dir_cur}/{augment_ind}.png')

