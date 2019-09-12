#coding=utf-8

from __future__ import print_function, division
# import os
# import argparse
import torch
import torch.nn as nn
from .model.cnn_geometric_model import CNNGeometric
from .image.normalization import NormalizeImageDict, normalize_image
from .geotnf.transformation import GeometricTnf
# from geotnf.point_tnf import *
from skimage import io
import warnings
# from torchvision.transforms import Normalize
from collections import OrderedDict

import numpy as np
from torch.autograd import Variable
import pdb

warnings.filterwarnings('ignore')

class GeoAPI(nn.Module):
    def __init__(self, feature_extraction_cnn='resnet101'):
        super(GeoAPI, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        self.resizeCNN = GeometricTnf(out_h=240, out_w=240).cuda()
        #normalizeTnf = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=self.use_cuda)
        self.affTnf = GeometricTnf(geometric_model='affine', use_cuda=self.use_cuda)

        self.feature_extraction_cnn = feature_extraction_cnn
        if feature_extraction_cnn == 'vgg':
            self.model_aff_path = 'trained_models/best_pascal_checkpoint_adam_affine_grid_loss.pth.tar'
            self.model_tps_path = 'trained_models/best_pascal_checkpoint_adam_tps_grid_loss.pth.tar'
        elif feature_extraction_cnn == 'resnet101':
            self.model_aff_path = 'trained_models/best_pascal_checkpoint_adam_affine_grid_loss_resnet_random.pth.tar'
            self.model_tps_path = 'trained_models/best_pascal_checkpoint_adam_tps_grid_loss_resnet_random.pth.tar'

        self.do_aff = not self.model_aff_path == ''
        self.do_tps = not self.model_tps_path == ''

        self.init_model()

    def init_model(self):
        # Create models
        print('Creating CNN models...')
        if self.do_aff:
            self.model_aff = CNNGeometric(use_cuda=self.use_cuda, geometric_model='affine', feature_extraction_cnn=self.feature_extraction_cnn)
        if self.do_tps:
            self.model_tps = CNNGeometric(use_cuda=self.use_cuda, geometric_model='tps', feature_extraction_cnn=self.feature_extraction_cnn)

        # Load trained weights
        print('Loading trained models weights...')
        if self.do_aff:
            checkpoint = torch.load(self.model_aff_path, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict(
                [(k.replace('vgg', 'models'), v) for k, v in checkpoint['state_dict'].items()])
            self.model_aff.load_state_dict(checkpoint['state_dict'])
        if self.do_tps:
            checkpoint = torch.load(self.model_tps_path, map_location=lambda storage, loc: storage)
            checkpoint['state_dict'] = OrderedDict(
                [(k.replace('vgg', 'models'), v) for k, v in checkpoint['state_dict'].items()])
            self.model_tps.load_state_dict(checkpoint['state_dict'])

        if self.do_aff:
            self.model_aff.eval()
        if self.do_tps:
            self.model_tps.eval()


    def preprocess_image(self, image):
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32) / 255.0)
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image_var = self.resizeCNN(image_var)

        # Normalize image
        image_var = normalize_image(image_var)

        return image_var

    ### source_image, target_image 为 parsing_a, parsing_b
    ### 如何传入中间层的future_map呢？
    def forward(self, source_image, target_image, a_image):
        # source_image_path = 'datasets/my_img/3-2.png'
        # target_image_path = 'datasets/my_img/3-1.png'
        # source_image = io.imread(source_image_path)
        # target_image = io.imread(target_image_path)

        resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda=self.use_cuda)

        ### 如果传入的本来就是正则化、var处理过的，则跳过
        source_image_var = self.preprocess_image(source_image)
        target_image_var = self.preprocess_image(target_image)
        a_image_var = self.preprocess_image(a_image)

        if self.use_cuda:
            source_image_var = source_image_var.cuda()
            target_image_var = target_image_var.cuda()
            a_image_var = a_image_var.cuda()

        batch = {'source_image': source_image_var, 'target_image': target_image_var}

        warped_image_aff, warped_image_tps, warped_image_aff_tps = None, None, None
        warped_image_aff_np, warped_image_tps_np, warped_image_aff_tps_np = None, None, None

        if self.do_aff:
            theta_aff = self.model_aff.forward(batch)
            # warped_image_aff = self.affTnf(batch['source_image'], theta_aff.view(-1, 2, 3))
            warped_image_aff = self.affTnf(a_image_var, theta_aff.view(-1, 2, 3))

        if self.do_tps:
            theta_tps = self.model_tps.forward(batch)
            # warped_image_tps = self.tpsTnf(batch['source_image'], theta_tps)
            warped_image_tps = self.tpsTnf(a_image_var, theta_tps)

        if self.do_aff and self.do_tps:
            theta_aff_tps = self.model_tps.forward({'source_image': warped_image_aff, 'target_image': batch['target_image']})
            warped_image_aff_tps = self.tpsTnf(warped_image_aff, theta_aff_tps)

        # Un-normalize images and convert to numpy
        if self.do_aff:
            warped_image_aff_np = normalize_image(resizeTgt(warped_image_aff), forward=False).data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()

        if self.do_tps:
            warped_image_tps_np = normalize_image(resizeTgt(warped_image_tps), forward=False).data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()

        if self.do_aff and self.do_tps:
            warped_image_aff_tps_np = normalize_image(resizeTgt(warped_image_aff_tps), forward=False).data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()

        return warped_image_aff_np, warped_image_tps_np, warped_image_aff_tps_np


    def get_thetas(self, source_image_path, target_image_path):

        source_image = io.imread(source_image_path)
        target_image = io.imread(target_image_path)
        pdb.set_trace()
        resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda=self.use_cuda)

        ### 如果传入的本来就是正则化、var处理过的，则跳过
        source_image_var = self.preprocess_image(source_image)
        target_image_var = self.preprocess_image(target_image)

        if self.use_cuda:
            source_image_var = source_image_var.cuda()
            target_image_var = target_image_var.cuda()

        batch = {'source_image': source_image_var, 'target_image': target_image_var}
        pdb.set_trace()
        theta_aff = self.model_aff.forward(batch)
        warped_image_aff = self.affTnf(batch['source_image'], theta_aff.view(-1, 2, 3))

        theta_tps = self.model_tps.forward(batch)
        # warped_image_tps = self.tpsTnf(batch['source_image'], theta_tps)

        theta_aff_tps = self.model_tps.forward({'source_image': warped_image_aff, 'target_image': batch['target_image']})
        # warped_image_aff_tps = self.tpsTnf(warped_image_aff, theta_aff_tps)

        return theta_aff, theta_tps, theta_aff_tps

    def get_thetas_from_image(self, source_image, target_image):
        # source_image_var = self.preprocess_image(source_image)
        # target_image_var = self.preprocess_image(target_image)
        # pdb.set_trace()
        source_image_var = self.resizeCNN(source_image)
        target_image_var = self.resizeCNN(target_image)
        
        source_image = source_image_var.cuda()
        target_image = source_image_var.cuda()

        resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda=self.use_cuda)        

        batch = {'source_image': source_image, 'target_image': target_image}

        theta_aff = self.model_aff.forward(batch)
        warped_image_aff = self.affTnf(batch['source_image'], theta_aff.view(-1, 2, 3))

        theta_tps = self.model_tps.forward(batch)
        theta_aff_tps = self.model_tps.forward({'source_image': warped_image_aff, 'target_image': batch['target_image']})

        return theta_aff, theta_tps, theta_aff_tps
























