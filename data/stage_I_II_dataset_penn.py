# coding=utf-8
import os
import json
import torch
import random
from data.base_dataset import BaseDataset
from utils import get_label_tensor, get_image_tensor, get_thetas_tensor, get_parsing_label_tensor, get_thetas_affgrid_tensor
from models.geo.geotnf.transformation import GeometricTnf


class Stage_I_II_Dataset_Penn(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.path_pairs = sorted(self.get_path_pairs(opt.pairs_path, opt.phase))
        self.path_pairs = self.get_path_pairs(opt.pairs_path, opt.phase)
        if not opt.serial_batches:
            random.shuffle(self.path_pairs)
            ### testing的时候设置了opt.serial_batches=True,所以random12800,得shuffle一下文件行不然都是前面的men了
        self.dataset_size = len(self.path_pairs)
        self.theta_json_data = json.load(open(opt.theta_json_path))

        self.tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=False)
        self.affTnf = GeometricTnf(geometric_model='affine', use_cuda=False)

    def __getitem__(self, index):
        a_jpg_path, b_jpg_path, a_parsing_path, b_json_path = self.get_paths(index)
        print (a_jpg_path, b_jpg_path, a_parsing_path, b_json_path)
        b_label_tensor, b_label_show_tensor = get_label_tensor(b_json_path, b_jpg_path, self.opt)

        a_parsing_tensor = get_parsing_label_tensor(a_parsing_path, self.opt)
        a_image_tensor = get_image_tensor(a_jpg_path, self.opt)
        b_image_tensor = get_image_tensor(b_jpg_path, self.opt)


        input_dict = {
            'a_image_tensor': a_image_tensor, \
            'b_image_tensor': b_image_tensor, \
            'b_label_tensor': b_label_tensor, \
            'a_parsing_tensor': a_parsing_tensor, \
            'b_label_show_tensor': b_label_show_tensor, \
            'a_jpg_path': a_jpg_path, \
            'b_jpg_path': b_jpg_path}

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
        a_jpg_path = p_list[0]
        b_jpg_path = p_list[1]

        a_parsing_path = a_jpg_path.replace('penn_action_my/', 'penn_action_my_parsing/')
        b_json_path = b_jpg_path.replace('.png', '_keypoints.json').replace('penn_action_my/', 'penn_action_my_keypoints/')

        print (a_jpg_path)
        print (b_jpg_path)
        print (a_parsing_path)
        print (b_json_path)
        return a_jpg_path, b_jpg_path, a_parsing_path, b_json_path


    def __len__(self):
        return len(self.path_pairs)

    def name(self):
        return 'Stage_I_II_Dataset_Penn'