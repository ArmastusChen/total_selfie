import torch
from torch import autocast
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2 
from torch.utils.data import Dataset
import torchvision.utils as vutils
import PIL.Image as Image
from misc import *
import numpy as np
import scipy.ndimage as ndimage
from skimage import morphology




def get_seg_by_name(seg, name):
    
    final_mask = np.zeros_like(seg)
    seg_inds = np.unique(seg)
    
    
    part_inds = PARTS[name]
    for seg_ind in seg_inds:
        if not seg_ind in part_inds:
            continue
        
        mask = seg_ind == seg 
        final_mask = final_mask + mask.astype(float)
        
    
    return final_mask.astype(float)




def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""
    
    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )


def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2


def dilate_mask(mask, dilate_iter):
    mask = ndimage.binary_dilation(mask, iterations=dilate_iter).astype(mask.dtype)
    return mask
    
def get_output_folder(config):
    path_config = config['path_config']
    base_config = config['base_config']
    train_config = config['train_config']
    num_inference_steps = base_config['num_inference_steps']
    dilate_iter = base_config['dilate_iter']
    shoes_dilate_iter = base_config['shoes_dilate_iter']

    
    control_path = path_config['control_path']


    control_name = control_path.split('/')[-1][:-4]

    use_scene_image = base_config['use_scene_image']
    scene_step = base_config['scene_step']
    use_soft_prompt_weights = base_config['use_soft_prompt_weights']
    controlnet_conditioning_scale = base_config['controlnet_conditioning_scale']

    scene_image = None

    if use_scene_image:
        scene_image_path = path_config['scene_path']
        scene_name = scene_image_path.split('/')[-1][:-4]
        control_name = f'{control_name}+{scene_name}+{scene_step}'
    
    control_name = f'{control_name}+{controlnet_conditioning_scale}+d{dilate_iter}_{shoes_dilate_iter}'

    out_folder = f"{control_name}/ep{num_inference_steps}_"

    if use_soft_prompt_weights:
        out_folder = f"{control_name}/soft_ep{num_inference_steps}_"

    use_first_n = train_config['use_first_n']
    prompts = base_config['prompts']
    prompt_steps = base_config['prompt_steps']
    prompt_weights = base_config['prompt_weights']

    prompt_steps = prompt_steps[:use_first_n]
    prompts = prompts[:use_first_n]
    prompt_weights = prompt_weights[:use_first_n]

    for i in range(len(prompts)):
        prompt = prompts[i]
        prompt_step = prompt_steps[i]
        prompt_weight = prompt_weights[i]
        out_folder += f"{prompt}_{prompt_step}+{prompt_weight}_"

    return out_folder




class VAE_dataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
     
        
        # self.names = os.listdir(self.images_dir)[:]
        self.paths = [images_dir]

        # print(self.names)


    def __len__(self):
        return len(self.paths)
    
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(f'{path}')
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = torch.from_numpy(img) / 127.5 - 1.0        
        img = img.permute(2, 0, 1)
        return {
            'images': img
        } 



def save_img_grid(image, path):
    pred_image = (image / 2 + 0.5).clamp(0, 1).float().cpu().detach()
    grid = vutils.make_grid(pred_image, nrow=8)

    vutils.save_image(grid, path)



def find_nearest_white(img, mask, tgt_mask):
    target_pixels = np.argwhere(abs(mask - tgt_mask) == 1)
    nonzero = np.argwhere(tgt_mask == 1)
    img_out = img.copy()
    count = 0
    for target_pixel in target_pixels:
        count +=1
        distances = np.sqrt((nonzero[:,0] - target_pixel[0]) ** 2 + (nonzero[:,1] - target_pixel[1]) ** 2)
        nearest_index = np.argmin(distances)
        nearest_cor = nonzero[nearest_index]
        img_out[target_pixel[0], target_pixel[1]] = img[nearest_cor[0], nearest_cor[1]]
    return img_out

def run_seg(image, ignore=False):
    from seg.demo import run_segment

    seg, _ = run_segment(image)
    

    scene_mask = get_seg_by_name(seg, 'scene') 
    
    person_mask = 1 - scene_mask
    person_mask = morphology.remove_small_objects(person_mask > 0, min_size=5).astype(np.float32)
    scene_mask = 1 - person_mask

    
    seg = person_mask * seg
    
    scene_mask_new = morphology.remove_small_objects(scene_mask > 0, min_size=1000).astype(np.float32)

    new_person_mask = 1 - scene_mask_new

    new_seg = find_nearest_white(seg, new_person_mask, person_mask)

    return np.uint8(new_seg), _

    
    



def compose_image_with_scene(image, scene_image, return_grid=False):

    image = np.array(image)
    scene_image = np.array(scene_image)
    
    img_seg, _ = run_seg(image)
    scene_mask = get_seg_by_name(img_seg, 'scene')[..., None]
    
    new_image = scene_image * scene_mask + image * (1-scene_mask)
    new_image = np.uint8(new_image)
    
    if return_grid:
        new_image = np.concatenate([image,new_image ], axis=1)
        
    new_image = Image.fromarray(new_image, mode="RGB")
    
    return new_image
    

    

def run_depth(img, normalize=True):
    
    img = Image.fromarray(img)
    
    from transformers import DPTImageProcessor, DPTForDepthEstimation

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    
    output = prediction.squeeze().cpu().numpy()
    

    if normalize:
        formatted = (output * 255 / np.max(output)).astype("uint8")
        formatted = formatted[..., None]
        formatted = np.concatenate([formatted, formatted, formatted], axis=2)

        formatted = Image.fromarray(formatted)
    else:
        formatted = output[..., None]
        formatted = np.concatenate([formatted, formatted, formatted], axis=2)

        
    return formatted



def run_normal(image):
    from transformers import pipeline

    img = Image.fromarray(image)
    
    depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )

    image = depth_estimator(img)['predicted_depth'][0]
  
    image = image.numpy()

    image_depth = image.copy()
    image_depth -= np.min(image_depth)
    image_depth /= np.max(image_depth)

    bg_threhold = 0.4

    x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    x[image_depth < bg_threhold] = 0

    y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    y[image_depth < bg_threhold] = 0

    z = np.ones_like(x) * np.pi * 2.0

    image = np.stack([x, y, z], axis=2)
    image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
    image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    
    image = image.resize((512,512), Image.BICUBIC)


    return image




def blur_with_mask(img, mask, sigma=3):
    from skimage.filters import gaussian

    img1 = gaussian(img * mask, sigma=sigma, channel_axis=-1)

    img2 = gaussian(mask, sigma=sigma, channel_axis=-1)

    blur_img = img1 / (img2+1e-6)
    blur_img = blur_img * mask

    return blur_img


def run_normal_bae(image):
    from transformers import pipeline
    from controlnet_aux import NormalBaeDetector

    image = Image.fromarray(image)
    processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

    image = processor(image)
    
    image = image.resize((512,512), Image.BICUBIC)


    return image



def dilate_mask(mask, dilate_size):
    mask = np.array(mask)
    mask[mask>128] = 255
    mask[mask<=128] = 0
    kernel = np.ones((int(dilate_size),int(dilate_size)),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = Image.fromarray(mask)

    return mask


def process_mask(mask_image, mask_type, mask_expand_ratio, res):
    if mask_type == 'rectangle':
        mask_image = np.array(mask_image)
        # find bbox and 
        bbox = np.argwhere(mask_image> 128)
        bbox = np.min(bbox, axis=0), np.max(bbox, axis=0)
        bbox = np.array(bbox)

        # expand bbox based on mask_expand_ratio
        bbox_size = bbox[1] - bbox[0]

        extra_bbox_size = bbox_size * (mask_expand_ratio - 1)
        extra_bbox_size = extra_bbox_size.astype(np.int32)
        bbox[0] = bbox[0] - extra_bbox_size // 2
        bbox[1] = bbox[1] + extra_bbox_size // 2
        bbox[0] = np.maximum(bbox[0], 0)
        bbox[1] = np.minimum(bbox[1], res)
        mask_image = np.zeros_like(mask_image)
        mask_image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]] = 255

        mask_image = Image.fromarray(mask_image)

    elif mask_type == 'dilate':
        mask_image = np.array(mask_image)
        mask_image[mask_image>128] = 255
        mask_image[mask_image<=128] = 0
        # dilate by 21 pixels
        kernel = np.ones((int(mask_expand_ratio),int(mask_expand_ratio)),np.uint8)
        mask_image = cv2.dilate(mask_image,kernel,iterations = 1)

        mask_image = Image.fromarray(mask_image)

    return mask_image