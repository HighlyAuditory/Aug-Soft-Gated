#coding=utf-8

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
# import sys
# sys.path.append('/home/disk2/donghaoye/ACMMM/semantic_align_gan_v3')

import torch
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from torch.autograd import Variable
from PIL import Image
from data.base_dataset import get_transform, get_params
from util.util import tensor2im, parsingim_2_tensor
from data.utils import get_image_tensor, get_parsing_label_tensor, get_label_tensor, get_thetas_tensor, get_thetas_affgrid_tensor


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

def get_test_result(a_jpg_path, b_jpg_path, model, opt):


    ### 1. 还要把val几个文件的JSON信息也储存到另外的文件，这样比直接读大JSON文件快。
    a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
    # b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')

    # a_json_path = a_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
    b_json_path = b_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
    src = a_jpg_path.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all/').split(os.sep)
    dst = b_jpg_path.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all/').split(os.sep)

    theta_pair_key = src[-2] + '_' + src[-1] + "=" + dst[-2] + '_' + dst[-1]
    b_parsing_label_filename = theta_pair_key.replace('=', '_TO_')
    b_parsing_label_filename = b_parsing_label_filename.replace('_vis.png', '') + '__fake_b_parsing.png'
    b_parsing_path = os.path.join(opt.joint_test_data_dir, b_parsing_label_filename)

    a_parsing_tensor = get_parsing_label_tensor(a_parsing_path, opt)
    b_parsing_tensor = get_parsing_label_tensor(b_parsing_path, opt)

    b_label_tensor, b_label_show_tensor = get_label_tensor(b_json_path, b_jpg_path, opt)
    a_image_tensor = get_image_tensor(a_jpg_path, opt)
    b_image_tensor = get_image_tensor(b_jpg_path, opt)

    a_parsing_rgb_tensor = parsingim_2_tensor(a_parsing_tensor, opt=opt, parsing_label_nc=opt.parsing_label_nc)
    b_parsing_rgb_tensor = parsingim_2_tensor(b_parsing_tensor, opt=opt, parsing_label_nc=opt.parsing_label_nc)

    theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor = get_thetas_affgrid_tensor(data_loader.dataset.affTnf, data_loader.dataset.tpsTnf, data_loader.dataset.theta_json_data, theta_pair_key)


    input_tensor = torch.cat([a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor, b_parsing_tensor, \
                              theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor], dim=0)
    input_var = Variable(input_tensor[None, :, :, :].type(torch.cuda.FloatTensor))
    model.eval()
    if opt.isTrain:
        fake_b = model.module.inference(input_var)
    else:
        fake_b = model.inference(input_var)

    return fake_b, a_image_tensor, b_image_tensor, b_label_show_tensor

def test_paper_img(tag='nips_cvpr'):
    web_dir = os.path.join(opt.results_dir, opt.name, "paper_image/" + tag, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    ### if test paper's image, then add save paper's result image
    root = "./datasets/deepfashion/paper_images/256"

    if tag == 'nips_cvpr':
        #path = "./datasets/deepfashion/paper_images/256/test_paper_img_path_joint_deepfashion.txt"
        path = "./datasets/deepfashion/paper_images/256/test_paper_img_path_joint.txt"
    elif tag == 'ablation':
        path = "./datasets/deepfashion/paper_images/256/test_paper_img_path_joint_ablation.txt"
        print (111111111111)
    else:
        path = "./datasets/deepfashion/paper_images/256/test_paper_img_path.txt"


    pairs = get_paper_testpairs(path)
    for i in xrange(len(pairs)):
        p = pairs[i]
        if tag == 'nips_cvpr' or tag == 'ablation':
            img_a = p[0]
            img_b = p[1]
            nips_result = p[2]
            cvpr_result = p[3]
            #paper_name = p[4]
        else:
            img_a = p[0]
            img_b = p[1]
            result = p[2]
            paper_name = p[3]

        a_jpg_path = os.path.join(root, img_a)
        b_jpg_path = os.path.join(root, img_b)

        fake_image, a_image_tensor, b_image_tensor, b_label_show_tensor = get_test_result(a_jpg_path, b_jpg_path, model, opt)

        test_list = [
            ('b_label_show', tensor2im(b_label_show_tensor)),
            ('a_image', tensor2im(a_image_tensor)),
            ('real_b_image', tensor2im(b_image_tensor)),
            ('fake_b_image', tensor2im(fake_image.data[0])) \
            #,
            #('fake_cloth', tensor2im(fake_cloth.data[0])),
            #('fake_head', tensor2im(fake_head.data[0]))
        ]

        ### paper result
        if tag == 'nips_cvpr' or tag == 'ablation':
            nips_result_path = os.path.join(root, nips_result)
            cvpr_result_path = os.path.join(root, cvpr_result)
            nips_result_tensor = get_paper_result(nips_result_path)
            cvpr_result_tensor = get_paper_result(cvpr_result_path)
            nips_result_tuple = ('nips_fake_b_image', tensor2im(nips_result_tensor))
            cvpr_result_tuple = ('cvpr_fake_b_image', tensor2im(cvpr_result_tensor))
            test_list.append(nips_result_tuple)
            test_list.append(cvpr_result_tuple)
        else:

            paper_result_path = os.path.join(root, result)
            paper_result_tensor = get_paper_result(paper_result_path)
            result_tuple = (paper_name + '_fake_b_image', tensor2im(paper_result_tensor))
            test_list.append(result_tuple)

        ### save image
        visuals = OrderedDict(test_list)
        visualizer.save_images(webpage, visuals, a_jpg_path, b_jpg_path)

        print('[%s]process image... %s' % (i, a_jpg_path))
    webpage.save()

def get_paper_result(old_result_path):
    old_result = Image.open(old_result_path).convert('RGB')
    params = get_params(opt, old_result.size)
    transform_image = get_transform(opt, params)
    old_result_tensor = transform_image(old_result)

    return old_result_tensor

def get_paper_testpairs(path):
    pairs = []
    lines = open(path).readlines()
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            pairs.append(line.split())

    return pairs

if __name__ == "__main__":
    flag = opt.which_img

    if flag == 'paper_img_nips_cvpr':
        test_paper_img(tag='nips_cvpr')
    elif flag == 'paper_img_nips':
        test_paper_img(tag='nips')
    elif flag == 'paper_img_ablation':
        test_paper_img(tag='ablation')
    else:
        print ('wrong which_img: ' + flag)



