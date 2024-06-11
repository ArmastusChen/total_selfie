# Data preprocessing, it resizes image to resxres, undistorts the face, crop the shoes region, infer the mask from the target pose image. 

import os
import sys 
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
import numpy as np 
from PIL import Image
import argparse
import glob


parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--skip_shoes', action='store_true')
parser.add_argument('--skip_face', action='store_true')
parser.add_argument('--skip_pose', action='store_true')
parser.add_argument('--res', type=int, default=512)
parser.add_argument('--gpu', type=str, default=None)
args = parser.parse_args()




name = args.name
res = args.res

root_dir = f'./data/{name}/raw'
dst_dir = f'./data/{name}/processed'

os.makedirs(dst_dir,exist_ok=True)


# process top 
top_path = glob.glob(f'{root_dir}/top*')[0]
top_dst_path = f'{dst_dir}/top.png'
top = Image.open(top_path).resize((res,res)).convert('RGB')
top.save(top_dst_path)


# process bottom
bottom_path = glob.glob(f'{root_dir}/bottom*')[0]
bottom_dst_path = f'{dst_dir}/bottom.png'
bottom = Image.open(bottom_path).resize((res,res)).convert('RGB')
bottom.save(bottom_dst_path)


# process scene 
scene_path = glob.glob(f'{root_dir}/scene*')[0]
scene_dst_path = f'{dst_dir}/scene.png'
scene = Image.open(scene_path).convert('RGB')
scene = scene.resize((res,res))
scene.save(scene_dst_path)



pose_path = glob.glob(f'{root_dir}/pose*')[0]
pose_dst_path = f'{dst_dir}/pose.png'
pose = Image.open(pose_path)
pose = pose.resize((res,res)).convert('RGB')
pose.save(pose_dst_path)



# process shoes
if not args.skip_shoes:
    shoes_path = glob.glob(f'{root_dir}/shoes*')[0]
    shoes_dst_path = f'{dst_dir}/shoes.png'

    shoes = Image.open(shoes_path).resize((res,res)).convert('RGB')
    from utils import run_seg, get_seg_by_name, get_crop_region, expand_crop_region

    
    # run the segmentation 
    seg, _ = run_seg(np.array(shoes))
    shoe_mask = get_seg_by_name(seg, 'shoes')
    crop_region = get_crop_region(shoe_mask)
    xmin, ymin, xmax, ymax = expand_crop_region(crop_region, res,res,res,res)

    # crop the shoes region 
    shoes = shoes.crop((xmin, ymin, xmax, ymax))
    shoes = shoes.resize((res,res))
    shoes.save(shoes_dst_path)

# process pose
if not args.skip_pose:
    from utils import run_seg, get_seg_by_name, get_crop_region, expand_crop_region

    # run the segmentation and get the mask of the person 
    seg, _ = run_seg(np.array(pose))
    pose_mask = 1 - get_seg_by_name(seg, 'scene')

    pose_mask = Image.fromarray(pose_mask.astype(np.uint8)*255)
    pose_mask = pose_mask.resize((res,res), Image.NEAREST)

    mask_dst_path = f'{dst_dir}/mask.png'

    pose_mask.save(mask_dst_path)

# process face
if not args.skip_face:
    face_path = glob.glob(f'{root_dir}/face*')[0]
    
    aligned_path  = f'{dst_dir}/face_a.png'

    # handle heic format if needed
    if face_path.endswith('.heic'):
        face = Image.open(face_path).resize((res,res)).convert('RGB')
        face_path = face_path.replace('.heic', '.png')
        face.save(face_path)


    face = Image.open(face_path).resize((res,res)).convert('RGB')

    # align the face to FFHQ format, this is the preprocessing step for the face correction model
    from ffhq_alignment import *
    face = run_alignment(np.array(face), res, res)
    face = Image.fromarray(face.astype(np.uint8)).convert('RGB').save(aligned_path)


    # call the face correction model
    cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python  face_correction/run_selfies_single.py --input_path {aligned_path}  --output_path {dst_dir}/face_corrected_a.png'
    os.system(cmd)










            

