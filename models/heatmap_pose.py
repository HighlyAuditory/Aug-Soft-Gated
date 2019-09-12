import argparse
import os
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import sys
import torch
from . import pose_estimation
import torch.nn.functional as F
import cv2

def preprocess(origin_img, image_size):
    normed_img = 0.5 * (origin_img - 0.5)
    height, width = image_size
    scale = 368.0 / height  # boxsize
    imgToTest = F.upsample(normed_img, scale_factor=(scale, scale), mode='bicubic')

    imgToTest_padded = torch.zeros([origin_img.shape[0], 3, 368, 256]).cuda()
    imgToTest_padded[:,:,:,:253] = imgToTest

    return imgToTest_padded

def process(model, input_var, mask_var):
    # get the features
    _, _, _, _, _, _, _, _, _, _, _, heatmap = model(input_var, mask_var)
    # (b, 19, 46, 32)
    heatmap = F.upsample(heatmap, scale_factor=(8, 8), mode='bicubic')
    heatmap = heatmap[..., :253]
    heatmap = F.upsample(heatmap, size=(256, 176), mode='bicubic')

    return heatmap[:,:18]

def construct_model(model_path):
    model = pose_estimation.PoseModel(num_point=19, num_vector=19)
    state_dict = torch.load(model_path)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    state_dict = model.state_dict()
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    return model
