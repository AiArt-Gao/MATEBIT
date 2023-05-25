# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random

import cv2
import torch
import numpy as np
from PIL import Image
from skimage import feature
from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform

class MetFacesDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(no_pairing_check=True)
        if is_train:
            parser.set_defaults(load_size=1024)
        else:
            parser.set_defaults(load_size=1024)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        # parser.set_defaults(label_nc=15)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        if opt.phase == 'train':
            fd = open('/data/home/scv8616/data/train.txt')
            lines = fd.readlines()
            fd.close()
        elif opt.phase == 'test':
            # celetometval
            fd = open('/data/home/scv8616/data/test.txt')
            lines = fd.readlines()
            fd.close()

        image_paths = []
        label_paths = []
        self.img_path = opt.dataroot
        self.mask_path = opt.maskroot
        for i in range(len(lines)):
            # id = random.randint(0, len(lines)) % len(lines)
            image_paths.append(os.path.join(self.img_path, lines[i].strip()))
            label_paths.append(os.path.join(self.mask_path, lines[i].strip()))
        return label_paths, image_paths

    def get_ref(self, opt):
        extra = ''
        # if opt.phase == 'test':
        #     extra = '_val'
        with open('/data/home/scv8616/data/metfaces_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else: # 取top5 5 和 1
                val = [items[1], items[2]]
            ref_dict[key] = val
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    # def get_edges(self, edge, t):
    #     edge[:,1:] = edge[:,1:] | (t[:,1:] != t[:,:-1])
    #     edge[:,:-1] = edge[:,:-1] | (t[:,1:] != t[:,:-1])
    #     edge[1:,:] = edge[1:,:] | (t[1:,:] != t[:-1,:])
    #     edge[:-1,:] = edge[:-1,:] | (t[1:,:] != t[:-1,:])
    #     return edge

    def get_label_tensor(self, path):

        img = Image.open(path).resize((self.opt.load_size, self.opt.load_size), resample=Image.BILINEAR).convert('RGB')
        params = get_params(self.opt, img.size)
        # inner_parts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'nose', 'u_lip', 'mouth', 'l_lip', 'eye_g', 'hair']
        # img_path = self.labelpath_to_imgpath(path)
        # img = Image.open(img_path).resize((self.opt.load_size, self.opt.load_size), resample=Image.BILINEAR)
        # params = get_params(self.opt, img.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # transform_img = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        label_tensor = transform_label(img)

        return label_tensor, params

    def imgpath_to_labelpath(self, path):
        # /data/haokang/MetFaces/metfaces/10075-00.png,
        img_root, name = path.split('metfaces/')
        label_path = os.path.join(self.mask_path, name)
        return label_path

    def labelpath_to_imgpath(self, path):

        met_root, name = path.split('edge/')
        img_path = os.path.join(self.img_path, name)
        return img_path


