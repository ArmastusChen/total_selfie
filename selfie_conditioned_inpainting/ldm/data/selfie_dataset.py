from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class SelfieImageDataset(data.Dataset):
    def __init__(self,state,arbitrary_mask_percent=0,**args
        ):
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            # A.HorizontalFlip(p=0.5),
            # A.Rotate(limit=20),
            # A.Blur(p=0.3),
            # A.ElasticTransform(p=0.3)
            ])
        dataset_dir = args['dataset_dir']
        self.dataset_dir = dataset_dir
        # self.bbox_path_list=[]


        self.data_list = []
        if state == "train":
            with open(f'{dataset_dir}/prompt.json', 'rt') as f:
                for line in f:
                    self.data_list.append(json.loads(line))
        elif state == "validation":
            with open(f'{dataset_dir}/prompt_val.json', 'rt') as f:
                for line in f:
                    self.data_list.append(json.loads(line))

    
        
        
        # self.data_list.sort()
        self.length=len(self.data_list)

        print(self.length)
 

    
    
    def __getitem__(self, index):
        # bbox_path=self.bbox_path_list[index]
        # file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.jpg'
        # dir_name=bbox_path.split('/')[-2]
        # img_path=os.path.join('dataset/open-images/images',dir_name,file_name)


        # bbox_list=[]
        # with open(bbox_path) as f:
        #     line=f.readline()
        #     while line:
        #         line_split=line.strip('\n').split(" ")
        #         bbox_temp=[]
        #         for i in range(4):
        #             bbox_temp.append(int(float(line_split[i])))
        #         bbox_list.append(bbox_temp)
        #         line=f.readline()
        # bbox=random.choice(bbox_list)


        data = self.data_list[index % self.length]

        source_filenames = data['source']
        target_filename = data['target']
        seg_filename = data['seg']

        img_path = os.path.join(self.dataset_dir, target_filename)


        img_p = Image.open(img_path).convert("RGB").resize((512, 512))

   
        ### Get reference image
        ref_image_tensors = []
        for source_filename in source_filenames[:]:
            ref_image_tensor = Image.open(f'{self.dataset_dir}/' + source_filename).convert("RGB")
            ref_image_tensor = np.array(ref_image_tensor)
            ref_image_tensor = self.random_trans(image=ref_image_tensor)
            ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
            ref_image_tensor=get_tensor_clip()(ref_image_tensor)

            ref_image_tensors.append(ref_image_tensor)

        ref_image_tensors = torch.cat(ref_image_tensors, dim=0)

        # print(ref_image_tensors.shape)


        # get bbox
        seg_image = Image.open(f'{self.dataset_dir}/' + seg_filename).convert("RGB")
        person_mask =  np.array(seg_image) > 0


        # get bbox from mask using np.where(person_mask)
        bbox = np.array(np.where(person_mask))
        bbox = [np.min(bbox[1]), np.min(bbox[0]), np.max(bbox[1]), np.max(bbox[0]), ]

        # print(bbox)
        # exit()


        # bbox_pad=copy.copy(bbox)
        # bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        # bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        # bbox_pad[2]=bbox[2]+min(10,img_p.size[0]-bbox[2])
        # bbox_pad[3]=bbox[3]+min(10,img_p.size[1]-bbox[3])
        # img_p_np=cv2.imread(img_path)
        # img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        # ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
        # ref_image_tensor=self.random_trans(image=ref_image_tensor)
        # ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        # ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]
        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))


        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255)) 
            bbox_mask=copy.copy(bbox)
            extended_bbox_mask=copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                            [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                        ])
            down_nodes = np.asfortranarray([
                    [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                    [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                ])
            left_nodes = np.asfortranarray([
                    [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                    [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                ])
            right_nodes = np.asfortranarray([
                    [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                    [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=5
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                        x_list.append(curve.evaluate(i*0.05)[0][0])
                        y_list.append(curve.evaluate(i*0.05)[1][0])
            mask_img_draw=ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image.fromarray(mask_img)
            mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)
        
        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize

        # print(image_tensor_resize.shape)
        # print(mask_tensor_resize.shape)
        # print(inpaint_tensor_resize.shape)



        # # save image_tensor_resize 
        # image_tensor_resize = image_tensor_resize.permute(1,2,0)
        # image_tensor_resize = image_tensor_resize.cpu().numpy()
        # image_tensor_resize = (image_tensor_resize + 1) / 2 * 255
        # image_tensor_resize = image_tensor_resize.astype(np.uint8)
        # image_tensor_resize = Image.fromarray(image_tensor_resize)
        # image_tensor_resize.save('image_tensor_resize.jpg')

        # # save inpaint_tensor_resize
        # inpaint_tensor_resize = inpaint_tensor_resize.permute(1,2,0)
        # inpaint_tensor_resize = inpaint_tensor_resize.cpu().numpy()
        # inpaint_tensor_resize = (inpaint_tensor_resize + 1) / 2 * 255
        # inpaint_tensor_resize = inpaint_tensor_resize.astype(np.uint8)
        # inpaint_tensor_resize = Image.fromarray(inpaint_tensor_resize)
        # inpaint_tensor_resize.save('inpaint_tensor_resize.jpg')

        # # save mask_tensor_resize
        # mask_tensor_resize = mask_tensor_resize[0]
        # mask_tensor_resize = mask_tensor_resize.cpu().numpy()
        # mask_tensor_resize = (mask_tensor_resize + 1) / 2 * 255
        # mask_tensor_resize = mask_tensor_resize.astype(np.uint8)
        # mask_tensor_resize = Image.fromarray(mask_tensor_resize)
        # mask_tensor_resize.save('mask_tensor_resize.jpg')
        
        # # save
        # for kk, ref_image_tensor in enumerate(ref_image_tensors):
        #     ref_image_tensor = ref_image_tensor.permute(1,2,0)
        #     ref_image_tensor = ref_image_tensor.cpu().numpy()
        #     # ref_image_tensor = (ref_image_tensor + 1) / 2 * 255
        #     ref_image_tensor = ref_image_tensor.astype(np.uint8)
        #     ref_image_tensor = Image.fromarray(ref_image_tensor)
        #     ref_image_tensor.save(f'ref_image_tensor_{kk}.jpg')
            


        

        # exit()
        

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensors}



    def __len__(self):
        return self.length



