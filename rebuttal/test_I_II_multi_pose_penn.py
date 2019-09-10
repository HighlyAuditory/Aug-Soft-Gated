#coding=utf-8

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys
# sys.path.append('/home/disk2/donghaoye/ACMMM/semantic_align_gan_v9')

import torch
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from torch.autograd import Variable
from util.util import tensor2im, parsingim_2_tensor

import cv2 as cv
from models.geo.geo_API import GeoAPI
from models.geo.generate_theta_json_20channel_baseon_src_dst_path import generate_theta
from models.geo.geotnf.transformation import GeometricTnf
from data.utils import get_thetas_affgrid_tensor_by_json, get_parsing_label_tensor, get_label_tensor

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

opt.stage = 123 ## choose stage_I_II_dataset.py
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

opt.name = "stage_I_gan_ganFeat_noL1_oneD_Parsing_bz50_parsing20_04222"
opt.which_G = "resNet"
opt.stage = 1
opt.which_epoch=100
model_1 = create_model(opt)

opt.name = "gan_L1_feat_vgg_notv_noparsing_afftps_05102228"
opt.which_G = "wapResNet_v3_afftps"
opt.stage = 2
opt.which_epoch=45
model_2 = create_model(opt)


visualizer = Visualizer(opt)
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

geo = GeoAPI()
affTnf = GeometricTnf(geometric_model='affine', use_cuda=False)
tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=False)


def main():
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.which_epoch, "I_and_II_pose_seq_penn_app"))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))



    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        a_image_tensor = data['a_image_tensor']         # 3
        b_image_tensor = data['b_image_tensor']         # 3
        # b_label_tensor = data['b_label_tensor']         # 18
        a_parsing_tensor = data['a_parsing_tensor']     # 1
        # b_label_show_tensor = data['b_label_show_tensor']
        a_jpg_path = data['a_jpg_path']
        b_jpg_path = data['b_jpg_path']

        a_parsing_rgb_tensor = parsingim_2_tensor(a_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)
        # show_image_tensor_1 = torch.cat((a_image_tensor, a_parsing_rgb_tensor), dim=3)
        show_image_tensor_1 = a_image_tensor

        test_list = []
        b_json_path_list = get_pose_seq_list(opt)
        for j in xrange(len(b_json_path_list)):
            b_json_path = b_json_path_list[j]
            b_label_tensor, b_label_show_tensor = get_label_tensor(b_json_path, b_jpg_path[0], opt)
            b_label_tensor = b_label_tensor.unsqueeze_(0)
            b_label_show_tensor = b_label_show_tensor.unsqueeze_(0)
            fake_b, fake_b_parsing_label_tensor = generate_fake_B(a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor)
            b_parsing_rgb_tensor = parsingim_2_tensor(fake_b_parsing_label_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)
            # show_image_tensor_1  = torch.cat((show_image_tensor_1, b_label_show_tensor, b_parsing_rgb_tensor, fake_b.data[0:1, :, :, :].cpu()), dim=3)
            show_image_tensor_1  = torch.cat((show_image_tensor_1, fake_b.data[0:1, :, :, :].cpu()), dim=3)

        test_list.append(('fake_image_seq', util.tensor2im(show_image_tensor_1[0])))

        ### save image
        visuals = OrderedDict(test_list)

        visualizer.save_images(webpage, visuals, a_jpg_path[0], b_jpg_path[0])

        if i % 1 ==0:
            print('[%s]process image... %s' % (i, a_jpg_path[0]))

    webpage.save()

    image_dir = webpage.get_image_dir()
    print( image_dir)


def get_pose_seq_list(opt):
    path_list = []
    lines = open(opt.pose_file_path).readlines()
    for line in lines:
        if len(line) < 1:
            break

        path = line.strip()
        # path = os.path.join(opt.dataroot, line.strip())
        # path = path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
        path_list.append(path)

    return path_list

def generate_fake_B(a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor):
    ##### stage I #################
    print (a_parsing_tensor.shape)
    print (b_label_tensor.shape)
    print ("################")
    input_1_tensor = torch.cat([a_parsing_tensor, b_label_tensor], dim=1)
    input_1_var = Variable(input_1_tensor.type(torch.cuda.FloatTensor))
    model_1.eval()
    fake_b_parsing = model_1.inference_2(input_1_var)

    a_parsing_tensor_RGB_numpy = util.parsing2im(
        util.label_2_onhot(a_parsing_tensor[0], parsing_label_nc=opt.parsing_label_nc))
    fake_b_parsing_RGB_numpy = util.parsing2im(fake_b_parsing.data[0])
    fake_b_parsing_label = util.parsing_2_onechannel(fake_b_parsing.data[0])

    a_parsing_RGB_path = './a_parsing_RGB.png'
    fake_b_parsing_RGB_path = './fake_b_parsing_RGB.png'
    fake_b_parsing_label_path = './fake_b_parsing_label.png'
    util.save_image(a_parsing_tensor_RGB_numpy, a_parsing_RGB_path)
    util.save_image(fake_b_parsing_RGB_numpy, fake_b_parsing_RGB_path)
    cv.imwrite(fake_b_parsing_label_path, fake_b_parsing_label)

    ##### GEO ######################

    theta_json = generate_theta(a_parsing_RGB_path, fake_b_parsing_RGB_path, geo)
    theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor = get_thetas_affgrid_tensor_by_json(affTnf, tpsTnf,
                                                                                                 theta_json)
    theta_aff_tensor = theta_aff_tensor.unsqueeze_(0)
    theta_tps_tensor = theta_tps_tensor.unsqueeze_(0)
    theta_aff_tps_tensor = theta_aff_tps_tensor.unsqueeze_(0)

    #### stage II #################
    fake_b_parsing_label_tensor = get_parsing_label_tensor(fake_b_parsing_label_path, opt)
    fake_b_parsing_label_tensor = fake_b_parsing_label_tensor.unsqueeze_(0)

    input_2_tensor = torch.cat([a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor, fake_b_parsing_label_tensor, \
         theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor], dim=1)
    input_2_var = Variable(input_2_tensor.type(torch.cuda.FloatTensor))
    model_2.eval()
    fake_b = model_2.inference(input_2_var)

    return fake_b, fake_b_parsing_label_tensor



if __name__ == "__main__":
    main()




