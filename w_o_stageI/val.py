#coding=utf-8
import os
import torch
from data.utils import get_image_tensor, get_parsing_label_tensor, get_label_tensor, get_thetas_tensor, get_thetas_affgrid_tensor
from util.util import tensor2im, parsingim_2_tensor
from torch.autograd import Variable



def get_test_result(a_jpg_path, b_jpg_path, model, opt, dataset):
    # a_json_path = a_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
    b_json_path = b_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')
    b_label_tensor, b_label_show_tensor = get_label_tensor(b_json_path, b_jpg_path, opt)
    a_image_tensor = get_image_tensor(a_jpg_path, opt)
    b_image_tensor = get_image_tensor(b_jpg_path, opt)

    input_tensor = torch.cat([a_image_tensor, b_image_tensor, b_label_tensor], dim=0)
    input_var = Variable(input_tensor[None, :, :, :].type(torch.cuda.FloatTensor))
    model.eval()
    if opt.isTrain:
        fake_b = model.module.inference(input_var)
    else:
        fake_b = model.inference(input_var)

    show_image_tensor = torch.cat((a_image_tensor, b_label_show_tensor, b_image_tensor, fake_b.data[0].cpu()), dim=2)

    return show_image_tensor

def get_valList(model, opt, dataset):
    val_list = []
    lines = open("./datasets/deepfashion/paper_images/256/val_img_path.txt").readlines()
    for i in range(len(lines)):
        image_a_path, image_b_path = lines[i].split()[0], lines[i].split()[1]
        a_jpg_path = os.path.join(opt.dataroot, image_a_path)
        b_jpg_path = os.path.join(opt.dataroot, image_b_path)

        show_image_tensor = get_test_result(a_jpg_path, b_jpg_path, model, opt, dataset)

        val_list.append(('val-a-b-fake-b-{}'.format(i), tensor2im(show_image_tensor)))

    return val_list

