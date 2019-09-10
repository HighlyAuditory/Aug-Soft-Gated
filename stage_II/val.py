#coding=utf-8
import os
import torch
from data.utils import get_image_tensor, get_parsing_label_tensor, get_label_tensor, get_thetas_tensor, get_thetas_affgrid_tensor
from util.util import tensor2im, parsingim_2_tensor
from torch.autograd import Variable



def get_test_result(a_jpg_path, b_jpg_path, model, opt, dataset):

    ### 1. 还要把val几个文件的JSON信息也储存到另外的文件，这样比直接读大JSON文件快。
    if 20 == opt.parsing_label_nc:
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

    theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor = get_thetas_affgrid_tensor(dataset.affTnf, dataset.tpsTnf, dataset.theta_json_data, theta_pair_key)


    input_tensor = torch.cat([a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor, b_parsing_tensor, \
                              theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor], dim=0)
    input_var = Variable(input_tensor[None, :, :, :].type(torch.cuda.FloatTensor))
    model.eval()
    if opt.isTrain:
        fake_b = model.module.inference(input_var)
    else:
        fake_b = model.inference(input_var)

    show_image_tensor_1 = torch.cat((a_image_tensor, b_label_show_tensor, b_image_tensor), dim=2)
    show_image_tensor_2 = torch.cat((a_parsing_rgb_tensor[0], b_parsing_rgb_tensor[0], fake_b.data[0].cpu()), dim=2)
    show_image_tensor = torch.cat((show_image_tensor_1, show_image_tensor_2), dim=1)

    return show_image_tensor

def get_valList(model, opt, dataset):
    val_list = []
    lines = open("./datasets/deepfashion/paper_images/256/val_img_path.txt").readlines()
    for i in xrange(len(lines)):
        image_a_path, image_b_path = lines[i].split()[0], lines[i].split()[1]
        a_jpg_path = os.path.join(opt.dataroot, image_a_path)
        b_jpg_path = os.path.join(opt.dataroot, image_b_path)

        show_image_tensor = get_test_result(a_jpg_path, b_jpg_path, model, opt, dataset)

        val_list.append(('val-a-b-fake-b-{}'.format(i), tensor2im(show_image_tensor)))

    return val_list

