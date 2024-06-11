import sys 
import os 
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
import PIL
import requests
import torch
from io import BytesIO
from pipelines.image_encoder import PaintByExampleImageEncoder
from PIL import Image
import numpy as np
import argparse
from misc import CACHE_DIR
from utils import blur_with_mask
import cv2 
from utils import run_seg, dilate_mask, process_mask



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--mask_type', type=str, default='rectangle')
parser.add_argument('--model_id', type=str, default=None)
parser.add_argument('--mask_expand_ratio', type=float, default=1.0)
parser.add_argument('--guidance_scale', type=float, default=5.0)
parser.add_argument('--controlnet_conditioning_scale', type=float, default=1.0)
parser.add_argument('--start_ind', type=int, default=0)
parser.add_argument('--end_ind', type=int, default=100)
parser.add_argument('--blending_step', type=int, default=30)
parser.add_argument('--mode', type=str, default=None )
parser.add_argument('--dilate_size', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--res', type=int, default=512)
parser.add_argument('--output_dir', type=str, default=None )


args = parser.parse_args()


mode = args.mode
blending_step = args.blending_step
name = args.name
mask_type = args.mask_type
mask_expand_ratio = args.mask_expand_ratio
guidance_scale = args.guidance_scale
model_id = args.model_id
dilate_size = args.dilate_size
controlnet_conditioning_scale = args.controlnet_conditioning_scale
model_name =  model_id.split('/')[-1]
res = args.res
end_ind = args.end_ind
start_ind = args.start_ind


root_dir = f'./data/{name}/processed'
output_dir = args.output_dir

# read the four selfies as input
parts = ['face_corrected_a', 'top', 'bottom', 'shoes']
example_images = [] 
for part in parts:
    example_images.append(Image.open(f'{root_dir}/{part}.png').convert('RGB')) 



# read pose image
pose_image = Image.open( f'{root_dir}/pose.png')

# read the image to be inpainted (background image)
init_image = Image.open(f'{root_dir}/scene.png').resize((res,res))
mask_image = Image.open(f'{root_dir}/mask.png').resize((res,res), Image.NEAREST)

mask_image_dilate = None
# dilate the mask
if dilate_size > 0:
    mask_image_dilate = dilate_mask(mask_image, dilate_size)
    

# prepare the inpainting mask 
mask_image = process_mask(mask_image, mask_type, mask_expand_ratio, res)



# initalize controlnet and prepare control image if needed
if mode is None:
    mode_name = 'no_cond'
    controlnet = None
    out_dir = f'{output_dir}/results/{mode_name}/{name}/{model_name}/{mask_type}/{mask_expand_ratio}/{guidance_scale}'
    os.makedirs(out_dir, exist_ok=True)

elif mode == 'pose':
    mode_name = 'pose'

    from diffusers import  ControlNetModel, EulerDiscreteScheduler
    from controlnet_aux import OpenposeDetector

    # prepare pose 
    model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", ).to('cuda')

    control_image  = pose_image


    out_dir = f'{output_dir}/results/{mode_name}/{name}/{model_name}/{mask_type}/{mask_expand_ratio}/{guidance_scale}'

    if dilate_size > 0:
        out_dir = f'{out_dir}_d_{dilate_size}'

    os.makedirs(out_dir, exist_ok=True)

    control_image = model(control_image,  hand_and_face=False, return_raw=False)
    control_image.save(f'{out_dir}/control.png')


    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", cache_dir=CACHE_DIR,torch_dtype=torch.float16, 
    ).to("cuda")


elif mode == 'canny':
    mode_name = 'canny'
    from diffusers import  ControlNetModel, EulerDiscreteScheduler

    
    low_threshold = 0.5
    high_threshold = 0.5


    seg, _  = run_seg(np.array(pose_image))
    

    # replace left hand with left arm, right hand with right arm, making sure not edge detected between hands and arms. 
    seg[seg==7] = 5
    seg[seg==8] = 6
    seg[seg==11] = 12
    seg[seg==34] = 12


    control_image = cv2.Canny(seg, low_threshold, high_threshold)
    control_image = control_image[:, :, None]
    control_image = np.concatenate([control_image, control_image, control_image], axis=2)
    control_image = Image.fromarray(control_image)

    out_dir = f'{output_dir}/results/{mode_name}/{name}/{model_name}/{mask_type}/{mask_expand_ratio}/{guidance_scale}'

    if dilate_size > 0:
        out_dir = f'{out_dir}_d_{dilate_size}'
    os.makedirs(out_dir, exist_ok=True)
    control_image.save(f'{out_dir}/control.jpg')

    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-canny", cache_dir=CACHE_DIR,torch_dtype=torch.float16
    ).to("cuda")


# read the pbe
if mode is None:
    from pipelines.pipeline_paint_by_example import PaintByExamplePipeline

    pipe = PaintByExamplePipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
    )

else:
    from pipelines.pipeline_paint_by_example_controlnet import PaintByExamplePipeline

    pipe = PaintByExamplePipeline.from_pretrained(
        model_id, 
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )


# prepare image encode
image_encoder = PaintByExampleImageEncoder.from_pretrained(f'{model_id}/image_encoder', torch_dtype=torch.float16,)
pipe.image_encoder = image_encoder
pipe = pipe.to("cuda")


# save input images as visualization
if dilate_size > 0:
    mask_image_dilate.save(f'{out_dir}/mask_dilate.png')

mask_image.save(f'{out_dir}/mask.png')
init_image.save(f'{out_dir}/scene.jpg')
pose_image.save(f'{out_dir}/pose.jpg')

for i, example_image in enumerate(example_images):
    example_image.save(f'{out_dir}/ex_{i}.jpg')


kwargs = {
        'blending_step': blending_step, 
        'mask_image_dilate': mask_image_dilate,
        }

if mode is not None:
    kwargs['control_image'] = control_image
    kwargs['controlnet_conditioning_scale'] = args.controlnet_conditioning_scale




if args.seed is not None:
    seed = args.seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(image=init_image, mask_image=mask_image, example_images=example_images, generator=generator,guidance_scale=guidance_scale, **kwargs).images[0]

    image.save(f'{out_dir}/{seed}.png')

else:
    for seed in range( start_ind, end_ind):
        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = pipe(image=init_image, mask_image=mask_image, example_images=example_images, generator=generator,guidance_scale=guidance_scale,  **kwargs).images[0]
        
        image.save(f'{out_dir}/{seed}.jpg')

    



