#coding=utf-8
import os
import random
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from utils import get_parsing_label_tensor, get_label_tensor
import pdb

class Stage_I_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print("opt.pairs_path={}".format(opt.pairs_path))
        pdb.set_trace()
        self.path_pairs = sorted(self.get_path_pairs(opt.pairs_path, opt.phase))
        if not opt.serial_batches:
            random.shuffle(self.path_pairs)
        self.dataset_size = len(self.path_pairs)
      
    def __getitem__(self, index):
        a_jpg_path, b_jpg_path, a_parsing_path, b_parsing_path, a_json_path, b_json_path =  self.get_paths(index)

        # a_parsing_tensor = get_parsing_tensor(a_parsing_path)
        # b_parsing_tensor = get_parsing_tensor(b_parsing_path)
        a_parsing_tensor = get_parsing_label_tensor(a_parsing_path, self.opt)
        # print("a_parsing_path={}, shape={}".format(a_parsing_path, a_parsing_tensor.shape))
        b_parsing_tensor = get_parsing_label_tensor(b_parsing_path, self.opt)
        # print("b_parsing_path={}, shape={}".format(b_parsing_path, b_parsing_tensor.shape))

        # a_label_tensor = get_label_tensor(a_json_path, a_jpg_path, self.opt)
        b_label_tensor, b_label_show_tensor = get_label_tensor(b_json_path, b_jpg_path, self.opt)

        input_dict = {'a_parsing_tensor': a_parsing_tensor,\
                      'b_parsing_tensor': b_parsing_tensor, \
                      'b_label_tensor': b_label_tensor, \
                      'b_label_show_tensor': b_label_show_tensor, \
                      'a_img_path': a_jpg_path, 'b_img_path': b_jpg_path}

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

        if 20 == self.opt.parsing_label_nc:
            a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
            b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
        elif 10 == self.opt.parsing_label_nc:
            a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all_10channel/')
            b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all_10channel/')

        a_json_path = a_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
        b_json_path = b_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')

        return a_jpg_path, b_jpg_path, \
               a_parsing_path, b_parsing_path, \
               a_json_path, b_json_path

    def __len__(self):
        return len(self.path_pairs)

    def name(self):
        return 'Stage_I_Dataset'