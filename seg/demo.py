# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0]))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from .m2fp import add_m2fp_config
from .predictor import VisualizationDemo

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
# print(sys.path)

# constants


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_m2fp_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=None, 
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config-file",
        default="configs/mhp-v2/m2fp_R101_bs16_145k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument('--grid_search', action='store_true', )
    parser.add_argument('--select_mode', action='store_true', )
    parser.add_argument(
        "--refs_folder",
        type=str,
        default='ref',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument('--use_selected_mode', action='store_true', )
    parser.add_argument('--face_only', action='store_true', )
    parser.add_argument('--ckpt_epoch',  type=int, default=None)
    parser.add_argument('--select_seed', type=int, default=5)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--start_ind', type=int, default=0)
    parser.add_argument('--end_ind', type=int, default=100)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'model_final.pth'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--skip_shoes', action='store_true')
    parser.add_argument('--skip_seg', action='store_true')
    parser.add_argument('--skip_face', action='store_true')
    parser.add_argument('--skip_gt', action='store_true')
    parser.add_argument('--use_align', action='store_true', )
    parser.add_argument('--use_inpaint', action='store_true', )
    parser.add_argument('--face_twice', action='store_true', )
    parser.add_argument('--use_landmark_warp', action='store_true', )
    parser.add_argument('--use_naive_inpaint', action='store_true', )
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--mask_type', type=str, default='rectangle')
    parser.add_argument('--mask_expand_ratio', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--selected_ind', type=int, default=5)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--blending_step', type=int, default=30)
    parser.add_argument('--use_y', action='store_true') 
    parser.add_argument('--dilate_blending', type=int, default=0)
    parser.add_argument('--controlnet_conditioning_scale', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default=None )
    parser.add_argument('--dilate_size', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument(
        "--refs_epoch",
        type=int,
        default=400,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument('--edit_epoch',  type=int, default=150)

    parser.add_argument('--crop_x', type=float, default=0.0)
    parser.add_argument('--crop_y', type=float, default=0.0)
    parser.add_argument('--crop_l', type=float, default=1.0)

    parser.add_argument('--crop_x_gt', type=float, default=0.0)
    parser.add_argument('--crop_y_gt', type=float, default=0.0)
    parser.add_argument('--crop_l_gt', type=float, default=1.0)

    parser.add_argument(
        "--control_type",
        type=str,
        default='depth',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument('--mode', type=str, default=None )
    parser.add_argument('--condition_path', type=str, default=None )
    parser.add_argument('--use_image_as_controlnet', action='store_true' )
    parser.add_argument('--parts', type=list, default=['face', 'shoes'], help='List of parts to be processed')
    parser.add_argument('--strengths', type=str, default=None, help='Comma-separated list of strengths for each part')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated list of selected final seeds')



    return parser



def run_segment(input_image):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))


    args.opts[1] = f'{cur_dir}/{args.opts[1]}'
    args.config_file = f'{cur_dir}/{args.config_file}'
    

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    args.output = 'result'

    # to bgr
    input_image = input_image[..., ::-1]

    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(input_image)

    pred = np.array(predictions["semantic_outputs"].argmax(dim=0).to('cpu'))
    # visualized_output = np.array(visualized_output)
    return pred, visualized_output

