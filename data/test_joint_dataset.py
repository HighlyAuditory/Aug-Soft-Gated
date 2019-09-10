# coding=utf-8
import os
import json
import torch
import random
from data.base_dataset import BaseDataset
from utils import get_label_tensor, get_image_tensor, get_thetas_tensor, get_parsing_label_tensor, get_thetas_affgrid_tensor
from models.geo.geotnf.transformation import GeometricTnf

### stage_I 和 stage_II联调，为了匹配出stage_I 输出的parsing label结果，作为Stage_II的test输入。
class TestJointDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.path_pairs = sorted(self.get_path_pairs(opt.pairs_path, opt.phase))
        if not opt.serial_batches:
            random.shuffle(self.path_pairs)
        self.dataset_size = len(self.path_pairs)
        self.theta_json_data = json.load(open(opt.theta_json_path))

        if not self.opt.no_dynamic_policy:
            self.policy_json_data = json.load(open(opt.policy_json_path))

        self.tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=False)
        self.affTnf = GeometricTnf(geometric_model='affine', use_cuda=False)

    def __getitem__(self, index):
        ### 这里暂时采用gt parsing b label, 后面再用根据parsing a + pose b生成的label
        a_jpg_path, b_jpg_path, a_parsing_path, b_parsing_path, a_json_path, b_json_path, theta_pair_key = self.get_paths(index)
        b_label_tensor, b_label_show_tensor = get_label_tensor(b_json_path, b_jpg_path, self.opt)

        a_parsing_tensor = get_parsing_label_tensor(a_parsing_path, self.opt)
        b_parsing_tensor = get_parsing_label_tensor(b_parsing_path, self.opt)
        a_image_tensor = get_image_tensor(a_jpg_path, self.opt)
        b_image_tensor = get_image_tensor(b_jpg_path, self.opt)

        theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor = get_thetas_affgrid_tensor(self.affTnf, self.tpsTnf, self.theta_json_data, theta_pair_key)  ### [1,18] --> 1*256*256
        if not self.opt.no_dynamic_policy:
            # policy_binary = get_policy_tensor(self.policy_json_data, theta_pair_key)  ### [1,18] --> 1*256*256
            print (1)
        # 固定policy方式跑四个模型, 后面再跑
        else:
            policy_binary = torch.zeros(1, a_parsing_tensor.size(1), a_parsing_tensor.size(2))
            if self.opt.which_policy == 'policy1': ### 00 (这个是必须跑的。事先给定00, 当为00时就代表已经去掉两个warp-block)
                policy_binary[:, 0, 0] = 0
                policy_binary[:, 0, 1] = 0
            elif self.opt.which_policy == 'policy2': ### 01
                policy_binary[:, 0, 0] = 0
                policy_binary[:, 0, 1] = 1
                #print policy_binary
            elif self.opt.which_policy == 'policy3': ### 10
                policy_binary[:, 0, 0] = 1
                policy_binary[:, 0, 1] = 0
            elif self.opt.which_policy == 'policy4': ### 11
                policy_binary[:, 0, 0] = 1
                policy_binary[:, 0, 1] = 1

        input_dict = {
            'a_image_tensor': a_image_tensor, \
            'b_image_tensor': b_image_tensor, \
            'b_label_tensor': b_label_tensor, \
            'a_parsing_tensor': a_parsing_tensor, \
            'b_parsing_tensor': b_parsing_tensor, \
            'b_label_show_tensor': b_label_show_tensor, \
            'theta_aff_tensor': theta_aff_tensor, \
            'theta_tps_tensor': theta_tps_tensor, \
            'theta_aff_tps_tensor': theta_aff_tps_tensor, \
            'policy_binary': policy_binary, \
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
        a_jpg_path = os.path.join(self.root, p_list[0])
        b_jpg_path = os.path.join(self.root, p_list[1])

        if 20 == self.opt.parsing_label_nc:
            a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
            #b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
        elif 10 == self.opt.parsing_label_nc:
            a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all_10channel/')
            #b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all_10channel/')

        a_json_path = a_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
        b_json_path = b_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')

        src = a_jpg_path.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all/').split(os.sep)
        dst = b_jpg_path.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all/').split(os.sep)

        # id_00006607_04_4_full_vis.png=id_00006607_04_1_front_vis.png
        theta_pair_key = src[-2] + '_' + src[-1] + "=" + dst[-2] + '_' + dst[-1]

        #"id_00000006_01_1_front_TO_id_00000006_01_2_side__fake_b_parsing.png"
        b_parsing_label_filename = theta_pair_key.replace('=', '_TO_')
        b_parsing_label_filename = b_parsing_label_filename.replace('_vis.png', '') + '__fake_b_parsing.png'
        b_parsing_path = os.path.join(self.opt.joint_test_data_dir, b_parsing_label_filename)

        return a_jpg_path, b_jpg_path, \
               a_parsing_path, b_parsing_path, \
               a_json_path, b_json_path, \
               theta_pair_key

    def __len__(self):
        return len(self.path_pairs)

    def name(self):
        return 'TestJointDataset'