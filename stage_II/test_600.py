#coding=utf-8

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys
import torch
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from torch.autograd import Variable

from PIL import Image
from data.base_dataset import get_transform, get_params
from util.util import tensor2im

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

def get_paper_testpairs(path):
    pairs = []
    lines = open(path).readlines()
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            pairs.append(line.split())

    return pairs

def get_paper_result(old_result_path):
    old_result = Image.open(old_result_path).convert('RGB')
    params = get_params(opt, old_result.size)
    transform_image = get_transform(opt, params)
    old_result_tensor = transform_image(old_result)

    return old_result_tensor

def test_paper_img(tag='nips_cvpr'):
    web_dir = os.path.join(opt.results_dir, opt.name, "paper_image", '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # print("-------------------------------------------webpage={}".format('Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch)))
    print("----------{}".format(web_dir))
    ### if test paper's image, then add save paper's result image
    root = "./datasets/deepfashion/paper_images/256"

    if tag == 'nips_cvpr':
        #path = "./datasets/deepfashion/paper_images/256/test_paper_img_path_joint_deepfashion.txt"
        path = "./datasets/deepfashion/paper_images/256/test_paper_img_path_joint.txt"
    elif tag == 'ablation':
        path = "./datasets/deepfashion/paper_images/256/test_paper_img_path_joint_ablation.txt"
    else:
        path = "./datasets/deepfashion/paper_images/256/test_paper_img_path.txt"


    pairs = get_paper_testpairs(path)
    for i in range(len(pairs)):
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

        fake_image, a_image_tensor, b_image_tensor, b_label_show_tensor =  get_test_result(a_jpg_path, b_jpg_path, model, opt)

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


def write_pairs(pairs, paper_tag):
    root_data = opt.dataroot

    web_dir = os.path.join(opt.results_dir, opt.name, paper_tag, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    for i in range(len(pairs)):
        p = pairs[i]
        img_a = p[0]
        img_b = p[1]
        paper_result_path = p[2]
        paper_name = p[3]

        a_jpg_path = os.path.join(root_data, img_a)
        b_jpg_path = os.path.join(root_data, img_b)

        a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')

        src = a_jpg_path.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all/').split(os.sep)
        dst = b_jpg_path.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all/').split(os.sep)
        theta_pair_key = src[-2] + '_' + src[-1] + "=" + dst[-2] + '_' + dst[-1]
        b_parsing_label_filename = theta_pair_key.replace('=', '_TO_')
        b_parsing_label_filename = b_parsing_label_filename.replace('_vis.png', '') + '__fake_b_parsing.png'
        b_parsing_path = os.path.join(opt.joint_test_data_dir, b_parsing_label_filename)

        if not os.path.exists(a_parsing_path) or not os.path.exists(b_parsing_path):
            continue

        fake_image, a_image_tensor, b_image_tensor, b_label_show_tensor = get_test_result(a_jpg_path, b_jpg_path, model, opt)


        test_list = [
            ('b_label_show', tensor2im(b_label_show_tensor)),
            ('a_image', tensor2im(a_image_tensor)),
            ('real_b_image', tensor2im(b_image_tensor)),
            ('fake_b_image', tensor2im(fake_image.data[0])) \
        ]

        ### paper result
        paper_result_tensor = get_paper_result(paper_result_path)
        result_tuple = (paper_name + '_fake_b_image', tensor2im(paper_result_tensor))
        test_list.append(result_tuple)

        ### save image
        visuals = OrderedDict(test_list)
        visualizer.save_images(webpage, visuals, a_jpg_path, b_jpg_path)
        print('[%s]process image... %s' % (i, a_jpg_path))

        webpage.save()


def get_com_id():
    cvpr18_set = set()
    cvpr18 = './datasets/paper_img_600/CVPR18_deepfashion_x_x_target_pair_paths.txt'
    for line in open(cvpr18).readlines():
        cvpr18_set.add(line.split()[0].split('/')[3])

    nips17_set = set()
    nips17 = './datasets/paper_img_600/NIP17_deepfashion_x_x_target_pair_paths.txt'
    for line in open(nips17).readlines():
        nips17_set.add(line.split()[0].split('/')[3])

    all_set = set()
    all = './datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_mvoneId_0118.txt'
    for line in open(all).readlines():
        if line.split()[-1] == 'test':
            all_set.add(line.split()[0].split('/')[3])

    tmp_set_cvpr = all_set & cvpr18_set
    tmp_set_nips = all_set & nips17_set

    return tmp_set_cvpr, tmp_set_nips


def test_600_imgs(flag):

    tag = flag.split('_')[-1]

    if tag == 'dp':
        path_cvpr18_dp = "./datasets/paper_img_600/CVPR18_deepfashion_x_x_target_pair_paths.txt"
        path_nip17_dp = "./datasets/paper_img_600/NIP17_deepfashion_x_x_target_pair_paths.txt"
        pairs_cvpr18_dp = get_paper_testpairs(path_cvpr18_dp)
        pairs_nip17_dp = get_paper_testpairs(path_nip17_dp)

        tmp_set_cvpr, tmp_set_nips = get_com_id()

        pairs_cvpr18_dp_tmp = []
        for i in range(len(pairs_cvpr18_dp)):
            id = pairs_cvpr18_dp[i][0].split('/')[3]
            print id
            if id in tmp_set_cvpr:
                pairs_cvpr18_dp_tmp.append(pairs_cvpr18_dp[i])

        pairs_nip17_dp_tmp = []
        for i in range(len(pairs_nip17_dp)):
            id = pairs_nip17_dp[i][0].split('/')[3]
            print id
            if id in tmp_set_nips:
                pairs_nip17_dp_tmp.append(pairs_nip17_dp[i])

        print "------------------------"
        print len(pairs_cvpr18_dp_tmp)  # 136
        print len(pairs_nip17_dp_tmp)   # 149
        write_pairs(pairs_cvpr18_dp_tmp, paper_tag='cvpr18_dp')
        write_pairs(pairs_nip17_dp_tmp, paper_tag='nip17_dp')

    elif tag == 'mk':
        path_cvpr18_mk = "./datasets/paper_img_600/CVPR18_market_x_x_target_pair_paths.txt"
        path_nip17_mk = "./datasets/paper_img_600/NIP17_market_x_x_target_pair_paths.txt"
        pairs_cvpr18_mk = get_paper_testpairs(path_cvpr18_mk)
        pairs_nip17_mk = get_paper_testpairs(path_nip17_mk)
        write_pairs(pairs_cvpr18_mk, paper_tag='cvpr18_mk')
        write_pairs(pairs_nip17_mk, paper_tag='nip17_mk')
    else:
        print ("wrong tag!!!!")




def test_paper_img_nips_cvpr_mk():
    web_dir = os.path.join(opt.results_dir, opt.name, "paper_image", '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    print("webpage={}".format(webpage))
    ### if test paper's image, then add save paper's result image
    root = opt.dataroot
    path = "./datasets/deepfashion/paper_images/256/test_paper_img_path_joint_market1501.txt"


    pairs = get_paper_testpairs(path)
    for i in range(len(pairs)):
        p = pairs[i]
        img_a = p[0]
        img_b = p[1]
        nips_result_path = p[2]
        cvpr_result_path = p[3]
        #paper_name = p[4]

        a_jpg_path = os.path.join(root, img_a)
        b_jpg_path = os.path.join(root, img_b)

        fake_image, a_image_tensor, b_image_tensor, b_label_show_tensor = get_test_result(a_jpg_path, b_jpg_path, model, opt)

        test_list = [
            ('b_label_show', tensor2im(b_label_show_tensor)),
            ('a_image', tensor2im(a_image_tensor)),
            ('real_b_image', tensor2im(b_image_tensor)),
            ('fake_b_image', tensor2im(fake_image.data[0]))
        ]

        nips_result_tensor = get_paper_result(nips_result_path)
        cvpr_result_tensor = get_paper_result(cvpr_result_path)
        nips_result_tuple = ('nips_fake_b_image', tensor2im(nips_result_tensor))
        cvpr_result_tuple = ('cvpr_fake_b_image', tensor2im(cvpr_result_tensor))
        test_list.append(nips_result_tuple)
        test_list.append(cvpr_result_tuple)

        ### save image
        visuals = OrderedDict(test_list)
        visualizer.save_images(webpage, visuals, a_jpg_path, b_jpg_path)

        print('[%s]process image... %s' % (i, a_jpg_path))
    webpage.save()




def main():
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    print("web_dir={}".format(web_dir))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    print("webpage={}".format(webpage))
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
        #if i > opt.how_many:
            break

        input_var = Variable(data['input_tensors'].type(torch.cuda.FloatTensor))
        a_image_tensor = data['a_image_tensor']
        b_image_tensor = data['b_image_tensor']
        b_label_show_tensor = data['b_label_show_tensor']
        a_img_path = data['a_img_path']
        b_img_path = data['b_img_path']

        fake_image = model.inference(input_var)

        test_list = [('b_label_show', util.tensor2im(b_label_show_tensor[0])),
                      ('a_image', util.tensor2im(a_image_tensor[0])),
                      ('fake_image', util.tensor2im(fake_image.data[0])),
                      ('b_image', util.tensor2im(b_image_tensor[0]))]

        ### save image
        visuals = OrderedDict(test_list)
        visualizer.save_images(webpage, visuals, a_img_path[0], b_img_path[0])

        print('[%s]process image... %s' % (i, a_img_path[0]))
        ### 从零开始为啥只有12779张？本来12800的！难道有11pair是重复的？检查pair文件。。
        ### 奇怪哦！难道要12800 + 21

    webpage.save()

    image_dir = webpage.get_image_dir()
    print image_dir
    # /results/mGPU_nofusion_noD2D3_lightCNN_tv_corr_sia_block3_bz6_0115/test_latest/images



if __name__ == "__main__":

    flag = opt.which_img

    if flag == 'all':
        main()
    elif flag == 'paper_img':
        test_paper_img(tag='nips_cvpr')
    elif flag.find('600_img') > -1:  ### 600_img_dp    600_img_mk
        test_600_imgs(flag)
    elif flag == 'nips_cvpr_mk':
        test_paper_img_nips_cvpr_mk()
    # elif flag == 'ablation':              ### 用test_my_paper_image.py做
    #     test_paper_img(tag='ablation')
    else:
        print ('wrong which_img: ' + flag)


    ### IS \ SSIM要单独运算，一起跑，会报显存

