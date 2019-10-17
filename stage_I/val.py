#coding=utf-8
import os
import torch
from data.utils import get_parsing_label_tensor, get_label_tensor
from util.util import tensor2im, parsing2im, label_2_onhot
from torch.autograd import Variable
import pdb

def get_test_result(a_jpg_path, b_jpg_path, model, opt):
    if 20 == opt.parsing_label_nc:
        a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
        b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all/')
    elif 10 == opt.parsing_label_nc:
        a_parsing_path = a_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all_10channel/')
        b_parsing_path = b_jpg_path.replace('.jpg', '.png').replace('img/', 'img_parsing_all_10channel/')

    b_json_path = b_jpg_path.replace('.jpg', '_keypoints.json').replace('img/', 'img_keypoint_json/')

    # a_parsing_tensor = get_parsing_tensor(a_parsing_path)
    # b_parsing_tensor = get_parsing_tensor(b_parsing_path)
    a_parsing_tensor = get_parsing_label_tensor(a_parsing_path, opt)
    b_parsing_tensor = get_parsing_label_tensor(b_parsing_path, opt)

    b_label_tensor, b_label_show_tensor,_ = get_label_tensor(b_json_path, b_jpg_path, opt)
    
    input_dict = {
            'b_label_tensor': b_label_tensor, \
            'a_parsing_tensor': a_parsing_tensor, \
            'b_parsing_tensor': b_parsing_tensor, \
            'b_label_show_tensor': b_label_show_tensor}

    # input_tensors = torch.cat((a_parsing_tensor, b_parsing_tensor, b_label_tensor), dim=0)
    # input_var = Variable(input_tensors[None, :, :, :].type(torch.cuda.FloatTensor))  ##torch.FloatTensor of size (1,34,256,256)
    # pdb.set_trace()
    model.eval()
    if opt.isTrain:
        fake_b_parsing = model.module.inference(input_dict)
    else:
        fake_b_parsing = model.inference(input_var)

    return a_parsing_tensor, b_parsing_tensor, fake_b_parsing, b_label_show_tensor

def get_valList(model, opt):
    val_list = []
    lines = open("./datasets/deepfashion/paper_images/256/val_img_path.txt").readlines()
    for i in range(len(lines)):
        image_a_path, image_b_path = lines[i].split()[0], lines[i].split()[1]
        a_jpg_path = os.path.join(opt.dataroot, image_a_path)
        b_jpg_path = os.path.join(opt.dataroot, image_b_path)
        a_parsing_tensor, b_parsing_tensor, fake_b_parsing, b_label_show_tensor = get_test_result(a_jpg_path, b_jpg_path, model, opt)

        val_list.append(('val_b_label_show_{}'.format(i), tensor2im(b_label_show_tensor)))
        val_list.append(('val_a_parsing_{}'.format(i), parsing2im(label_2_onhot(a_parsing_tensor))))
        val_list.append(('val_b_parsing_{}'.format(i), parsing2im(label_2_onhot(b_parsing_tensor))))
        val_list.append(('val_fake_b_parsing_{}'.format(i), parsing2im(fake_b_parsing.data[0])))

    return val_list

