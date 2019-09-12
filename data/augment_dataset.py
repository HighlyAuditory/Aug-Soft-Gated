# coding=utf-8
import os
import json
import torch
import random
import numpy as np
import pdb

from data.base_dataset import BaseDataset
from .utils import get_label_tensor, get_image_tensor, get_parsing_label_tensor
from models.geo.geotnf.transformation import GeometricTnf
"""
{ 
Source: 3d angle skeleton of the 2d projection.
    format: [x, y, depth]
Target: unsupervised 3d skeleton with its 2d projection
The projection direction is depth axis as in 2d->3d}
The 3d skeleton should be in format of (x, y, depth)
"""
class Augment_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.path_pairs = self.get_path_pairs(opt.pairs_path, opt.phase)
        
        print("_-------pairs path={}".format(opt.pairs_path))
        if not opt.serial_batches:
            random.shuffle(self.path_pairs)
            ### testing的时候设置了opt.serial_batches=True,所以random12800,得shuffle一下文件行不然都是前面的men了
        self.dataset_size = len(self.path_pairs)
        self.dir_json = os.path.join('/home/wenwens/Documents/HumanPose/Pose-Transfer-pose3d-normed/fashion_data', opt.phase + '_3d_top_ordered') #keypoints

    def __getitem__(self, index):
        # \wenwen{need rewrite after generate all training pairs}
        a_jpg_path, b_jpg_path, a_parsing_path, b_parsing_path, b_json_path, a_json_path = self.get_paths(index)
        K1_path = os.path.join(self.dir_json, a_jpg_path.replace('.jpg','.npy'))
        K2_path = os.path.join(self.dir_json, b_jpg_path.replace('.jpg','.npy'))

        Kd1, Kd2 = np.load(K1_path,allow_pickle=True).item(), np.load(K2_path,allow_pickle=True).item()
        K1, K2 = Kd1['absolute_angles'], Kd2['absolute_angles']
        L2, F2 = Kd2['limbs'], Kd2['offset'].squeeze()
        L1, F1 = Kd1['limbs'], Kd1['offset'].squeeze()

        b_label_tensor, b_label_show_tensor = get_label_tensor(b_json_path, b_jpg_path, self.opt)
        a_label_tensor, _ = get_label_tensor(a_json_path, a_jpg_path, self.opt)

        a_parsing_tensor = get_parsing_label_tensor(a_parsing_path, self.opt)
        b_parsing_tensor = get_parsing_label_tensor(b_parsing_path, self.opt)
        a_image_tensor = get_image_tensor(a_jpg_path, self.opt)
        b_image_tensor = get_image_tensor(b_jpg_path, self.opt)

        input_dict = {
            'a_image_tensor': a_image_tensor, \
            'b_image_tensor': b_image_tensor, \
            'b_label_tensor': b_label_tensor, \
            'a_label_tensor': a_label_tensor, \
            'K1': K1, 'K2': K2,'L2':L2, 'F2': F2, 'L1':L1, 'F1': F1,
            'a_parsing_tensor': a_parsing_tensor, \
            'b_parsing_tensor': b_parsing_tensor, \
            'b_label_show_tensor': b_label_show_tensor, \
            'a_jpg_path': a_jpg_path}

        return input_dict

    def get_path_pairs(self, pairs_path, phase):
        p = []
        lines = open(pairs_path).readlines()
        for l in lines:
            if l.split()[-1].strip() == phase:
                p.append(l)

        return p

    def get_paths(self, index):
        line = self.path_pairs[index]
        p_list = line.split()
        a_jpg_path = os.path.join(self.root, p_list[0])
        b_jpg_path = os.path.join(self.root, p_list[1])

        a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
        b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
        b_json_path = b_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
        a_json_path = a_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')

        return a_jpg_path, b_jpg_path, a_parsing_path, b_parsing_path, b_json_path, a_json_path


    def __len__(self):
        return len(self.path_pairs)

    def name(self):
        return 'Augment_Dataset'