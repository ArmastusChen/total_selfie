import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from pipelines.stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
import yaml
from misc import *
from utils import *
import cv2
import argparse


parser = argparse.ArgumentParser(description='Inpainting')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--mask_type', type=str, default='rectangle')
parser.add_argument('--mask_expand_ratio', type=float, default=1.0)
parser.add_argument('--guidance_scale', type=float, default=5.0)
parser.add_argument('--selected_ind', type=int, default=5)
parser.add_argument('--edit_epoch',  type=int, default=150)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--ckpt_epoch', type=int, default=800)
parser.add_argument('--res', type=int, default=512)
parser.add_argument('--dilate_size', type=int, default=0)
parser.add_argument('--output_dir', type=str, default=None )
parser.add_argument('--model_id', type=str, default=None)
parser.add_argument('--parts', type=list, default=['face', 'shoes'], help='List of parts to be processed')
parser.add_argument('--strengths', type=str, default=None, help='Comma-separated list of strengths for each part')
parser.add_argument('--seeds', type=str, default=None, help='Comma-separated list of selected final seeds')
parser.add_argument('--controlnet_conditioning_scale', type=float, default=0.5, help='ControlNet conditioning scale')

args = parser.parse_args()

# args
mode = args.mode
name = args.name
mask_type = args.mask_type
mask_expand_ratio = args.mask_expand_ratio
guidance_scale = args.guidance_scale
selected_ind = args.selected_ind
ckpt_epoch = args.ckpt_epoch
dilate_size = args.dilate_size
output_dir = args.output_dir
model_id = args.model_id
parts = args.parts
controlnet_conditioning_scale = args.controlnet_conditioning_scale
res = args.res
edit_epoch = args.edit_epoch

strengths = list(map(float, args.strengths.split(',')))
if args.seeds is not None:
    seeds = list(map(int, args.seeds.split(',')))
else:
    seeds = None

# get model name
model_name =  model_id.split('/')[-1]

# get folder name
folder_name = get_folder_name(name, model_name, mask_type, mask_expand_ratio, guidance_scale)

# get prompt
prompts = {
    'face': 'sks face',
    'shoes': 'hta shoes',
}

# get edit model path
edit_model_path = f'./{output_dir}/fine_tune_db_ckpt/{name}/{edit_epoch}' 

folder_name = folder_name.replace(model_name, f'checkpoint-{ckpt_epoch}')

# get output path
if seeds is not None:
    refine_out_path = f'{output_dir}/results_final/{mode}/{folder_name}'
else:
    refine_out_path = f'{output_dir}/results_refine/{mode}/{folder_name}'

strengths_str = '_'.join([str(strength) for strength in strengths])
refine_out_path = f'{refine_out_path}/{selected_ind}/{strengths_str}'
os.makedirs(refine_out_path, exist_ok=True)


# load images
if dilate_size > 0:
    init_output_path = f'{output_dir}/results/{mode}/{folder_name}_d_{dilate_size}/{selected_ind}.png'
else:
    init_output_path = f'{output_dir}/results/{mode}/{folder_name}/{selected_ind}.png'

init_output = load_image(init_output_path).resize((res, res))
init_output.save(f'{refine_out_path}/init_output.png')

# run segmentation 
seg, _ = run_seg(np.array(init_output), True)
seg = Image.fromarray(seg.astype(np.uint8))
seg.save(f'{refine_out_path}/seg.png')



# load pose image
root_dir = f'./data/{name}/processed'
pose_image = Image.open( f'{root_dir}/pose.png')

# run segmentation
seg_pose, _ = run_seg(np.array(pose_image), True)
seg_pose = Image.fromarray(seg_pose.astype(np.uint8))
seg_pose.save(f'{refine_out_path}/seg_pose.png')


# load controlnet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16, cache_dir=CACHE_DIR )



# create inpainting pipeline
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(edit_model_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

init_output_ori = init_output.copy()

for seed in range(100):
    refine_output = init_output_ori.copy()

    for i, part in enumerate(parts):
        if seeds is not None:
            seed = seeds[i]

        strength = strengths[i]
        mask = get_seg_by_name(seg, part)

        # gt mask 
        pose_mask = get_seg_by_name(seg_pose, part)

        # get bbox 
        crop_region = get_crop_region(mask)
        xmin, ymin, xmax, ymax = expand_crop_region(crop_region, res,res,res,res)

        # crop mask 
        mask = mask[ymin:ymax, xmin:xmax]
            
        # resize to height, width
        mask = cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)

        init_output_cur = refine_output.crop((xmin, ymin, xmax, ymax)).resize((res, res))

        # detect canny  edge
        low_threshold = 0.5
        high_threshold = 0.5
        cur_mask = mask
        control_image = cv2.Canny(np.uint8(cur_mask), low_threshold, high_threshold)
        control_image = Image.fromarray(control_image.astype(np.uint8)).convert('RGB')


        # dilate 
        kernel = np.ones((11,11),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert('RGB')

        # set generator seed
        generator = torch.Generator(device='cuda')
        generator.manual_seed(seed)

        num_images_per_prompt = 1
        kwargs = {
                'controlnet_conditioning_image': control_image,
                'image': [init_output_cur for j in range(num_images_per_prompt)],
                'mask_image': mask_image,
                'prompt':prompts[part],
                'num_inference_steps': 30,
                'generator': generator,
                'num_images_per_prompt': num_images_per_prompt,
                'strength': strength,
                'controlnet_conditioning_scale': controlnet_conditioning_scale, 
                'height': res,
                'width': res,
        }

        # run inpainting
        result_imgs = pipe(**kwargs).images

        result_img = result_imgs[0]


        # composite with init_output_cur
        result_img = np.array(result_img)

        if not part in ['face']:
            init_output_cur = np.array(init_output_cur)
            if len(mask.shape) == 2:
                mask = mask[..., None]
            result_img = result_img * mask + init_output_cur * (1 - mask)

            

        result_img = Image.fromarray(result_img.astype(np.uint8)).convert('RGB')

        # resize
        result_img = result_img.resize((xmax - xmin, ymax - ymin), Image.ANTIALIAS)

        refine_output.paste(result_img, (xmin, ymin, xmax, ymax))


        if seeds is not None:
            refine_output.save(f'{refine_out_path}/final.png')
        else:
            refine_output.save(f'{refine_out_path}/{seed}.jpg')



    if seeds is not None:
        exit()










