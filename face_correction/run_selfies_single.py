import matplotlib
matplotlib.use('Agg')
import os, sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback


# from modules.keypoint_detector import KPDetector, HEEstimator
from animate import normalize_kp
from scipy.spatial import ConvexHull
from ffhq_align import image_align
import cv2
from utils import *

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='./face_correction/config/eg3d.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='./face_correction/ckpt/00000196-checkpoint.pth.tar', help="path to checkpoint to restore")


    parser.add_argument("--input_path", default=None, help="path to source image")
    parser.add_argument("--output_path", default=None, help="path to source image")

    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")


    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--free_view", dest="free_view", action="store_true", help="control head pose")
    parser.add_argument("--yaw", dest="yaw", type=int, default=None, help="yaw")
    parser.add_argument("--pitch", dest="pitch", type=int, default=None, help="pitch")
    parser.add_argument("--roll", dest="roll", type=int, default=None, help="roll")
    parser.add_argument("--image_res", type=int, default=512, help="roll")
 

    parser.set_defaults(relative=True)
    parser.set_defaults(adapt_scale=True)
    parser.set_defaults(free_view=False)

    opt = parser.parse_args()
    
    input_path = opt.input_path
    output_path = opt.output_path


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_res = opt.image_res

    # load models
    generator, kp_detector, he_estimator, he_estimator_far = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, gen=opt.gen, cpu=opt.cpu)

    with open(opt.config) as f:
        config = yaml.load(f)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    

    source_image = cv2.imread(input_path)[..., ::-1]
    source_image = source_image / 255.

    
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.cuda()
        
        # feature extraction
        kp_canonical = kp_detector(source)
        he_source = he_estimator(source)
        he_driving = he_estimator_far(source)

        # transform keypoints
        kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
        kp_driving = keypoint_transformation(kp_canonical, he_driving, estimate_jacobian)

        # face correction
        out = generator(source, kp_source=kp_source, kp_driving=kp_driving)

    # post processing
    img = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
    
    # to uint 
    img = (img * 255).astype(np.uint8)

    # save the img to result_dir using name
    imageio.imwrite(output_path, img)
        
    
        
        
        