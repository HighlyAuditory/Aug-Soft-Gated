#coding=utf-8

import torch
import numpy as np
import cv2 as cv
from data.base_dataset import get_transform, get_params
from .draw_point_by_json import *
from util.util import parsing2im, CV2PIL
from PIL import Image
from torch.autograd import Variable

# def get_parsing_tensor(parsing_path):
def get_parsing_tensor(parsing_path, opt):
    parse = Image.open(parsing_path)

    # parsing_label_array = torch.zeros(9L, parse.size[0], parse.size[1])
    parsing_label_array = np.zeros((10, parse.size[0], parse.size[1]), dtype=np.float32)

    parse_array = np.array(parse)
    # parse_person = (parse_array > 0).astype(np.float32)
    parse_background = (parse_array <= 0).astype(np.float32)

    # parse_head = (parse_array == 13).astype(np.float32) + \
    #              (parse_array == 2).astype(np.float32) + \
    #              (parse_array == 1).astype(np.float32) + \
    #              (parse_array == 4).astype(np.float32)

    parse_hair = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32)

    parse_face = (parse_array == 4).astype(np.float32) + \
                 (parse_array == 13).astype(np.float32)

    parse_body_skin = (parse_array == 10).astype(np.float32)
    parse_up_cloth = (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32) + \
                     (parse_array == 11).astype(np.float32)

    parse_low_cloth = (parse_array == 9).astype(np.float32) + \
                      (parse_array == 12).astype(np.float32)

    parse_left_hand = (parse_array == 3).astype(np.float32) + \
                      (parse_array == 15).astype(np.float32)
    parse_right_hand = (parse_array == 3).astype(np.float32) + \
                       (parse_array == 14).astype(np.float32)

    parse_left_leg = (parse_array == 17).astype(np.float32) + \
                     (parse_array == 19).astype(np.float32) + \
                     (parse_array == 8).astype(np.float32)
    parse_right_leg = (parse_array == 16).astype(np.float32) + \
                      (parse_array == 18).astype(np.float32) + \
                      (parse_array == 8).astype(np.float32)

    parsing_label_array[0] = parse_background
    parsing_label_array[1] = parse_hair
    parsing_label_array[2] = parse_face
    parsing_label_array[3] = parse_body_skin
    parsing_label_array[4] = parse_up_cloth
    parsing_label_array[5] = parse_low_cloth
    parsing_label_array[6] = parse_left_hand
    parsing_label_array[7] = parse_right_hand
    parsing_label_array[8] = parse_left_leg
    parsing_label_array[9] = parse_right_leg

    # params = get_params(opt, parse.size)
    # transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)   ### 不用正则化
    # print parsing_label_array.shape
    # parsing_label_tensor = transform_label(parsing_label_array) * 255.0
    parsing_label_tensor = torch.from_numpy(parsing_label_array)

    return parsing_label_tensor

def get_parsing_label_tensor(parsing_path, opt):
    label = Image.open(parsing_path)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0

    return label_tensor

def get_label_tensor(json_path, jpg_path, opt):
    # 0 ~ 256
    image_a = Image.open(jpg_path).convert('RGB')
    params = get_params(opt, image_a.size)
    transform_image = get_transform(opt, params)
    points = get_points(json_path)
    label_18chnl_tensor = draw_18chnl_points(points, transform_image)
    ### b 18 point show
    label_show = draw_points(points)
    label_show_tensor = transform_image(label_show)

    return label_18chnl_tensor, label_show_tensor, points

def get_label_tensor_from_kpts(kpts, jpg_path, opt):
    params = get_params(opt, np.array([256, 256]))
    transform_image = get_transform(opt, params)
    label_show = draw_points(kpts)
    label_show_tensor = transform_image(label_show)

    return label_show_tensor

def get_parsing_parts(parsing_path, jpg_path, opt):
    image = Image.open(jpg_path).convert('RGB')
    params = get_params(opt, image.size)
    transform = get_transform(opt, params)

    parsing_label = get_parsing_tensor(parsing_path)
    parsing_image = parsing2im(parsing_label)
    parsing_image = CV2PIL(parsing_image)
    parsing_image = transform(parsing_image)

    parsing_label = Image.open(parsing_path)
    parsing_parts_tensor = cut_parts(parsing_label, parsing_image)

    return parsing_parts_tensor

def cut_parts(parsing, im):
    parse_array = np.array(parsing)

    #parse_person = (parse_array > 0).astype(np.float32)

    parse_background = (parse_array <= 0).astype(np.float32)

    # parse_head = (parse_array == 13).astype(np.float32) + \
    #              (parse_array == 2).astype(np.float32) + \
    #              (parse_array == 1).astype(np.float32) + \
    #              (parse_array == 4).astype(np.float32)

    parse_hair = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32)

    parse_face = (parse_array == 4).astype(np.float32) + \
                 (parse_array == 13).astype(np.float32)

    parse_body_skin = (parse_array == 10).astype(np.float32)

    parse_up_cloth = (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32) + \
                     (parse_array == 11).astype(np.float32)

    parse_low_cloth = (parse_array == 9).astype(np.float32) + \
                      (parse_array == 12).astype(np.float32)

    parse_left_hand = (parse_array == 3).astype(np.float32) + \
                      (parse_array == 15).astype(np.float32)
    parse_right_hand = (parse_array == 3).astype(np.float32) + \
                       (parse_array == 14).astype(np.float32)

    parse_left_leg = (parse_array == 17).astype(np.float32) + \
                      (parse_array == 19).astype(np.float32) + \
                      (parse_array == 8).astype(np.float32)
    parse_right_leg = (parse_array == 16).astype(np.float32) + \
                     (parse_array == 18).astype(np.float32) + \
                     (parse_array == 8).astype(np.float32)

    # head = cut(im, parse_array, parse_head)
    hair = cut(im, parse_array, parse_hair)
    face = cut(im, parse_array, parse_face)
    body_skin = cut(im, parse_array, parse_body_skin)
    up_cloth = cut(im, parse_array, parse_up_cloth)
    low_cloth = cut(im, parse_array, parse_low_cloth)
    left_hand = cut(im, parse_array, parse_left_hand)
    right_hand = cut(im, parse_array, parse_right_hand)
    left_leg = cut(im, parse_array, parse_left_leg)
    right_leg = cut(im, parse_array, parse_right_leg)

    # person = cut(im, parse_array, parse_person)
    background = cut(im, parse_array, parse_background)

    ### 利用身体对称性，补全左(右)手(脚)
    if sum(sum(parse_left_hand)) == 0:
        left_hand = right_hand
    if sum(sum(parse_right_hand)) == 0:
        right_hand = left_hand
    if sum(sum(parse_left_leg)) == 0:
        left_leg = right_leg
    if sum(sum(parse_right_leg)) == 0:
        right_leg = left_leg

    parts = torch.cat((hair, face, body_skin, up_cloth, low_cloth, left_hand, right_hand, left_leg, right_leg, background), dim=0)

    return parts


def cut(im, parse_array, parse_head):
    parse_head = np.tile(parse_head, (3, 1))
    parse_head = parse_head.reshape((3, parse_array.shape[0], parse_array.shape[1]))
    mask_head = torch.from_numpy(parse_head)
    head = im * mask_head + mask_head - 1
    head = head.type(torch.FloatTensor)

    return head


def get_parsing_foreground(parsing_path, jpg_path, opt):
    image = Image.open(jpg_path).convert('RGB')
    params = get_params(opt, image.size)
    transform = get_transform(opt, params)
    image = transform(image)

    parsing_label = get_parsing_tensor(parsing_path)
    parsing_image = parsing2im(parsing_label)
    parsing_image = CV2PIL(parsing_image)
    parsing_image = transform(parsing_image)

    parsing_label = Image.open(parsing_path)
    parsing_parts_tensor = cut_forground(parsing_label, parsing_image)
    image_parts_tensor = cut_forground(parsing_label, image)

    return parsing_parts_tensor, image_parts_tensor



def cut_forground(parsing, im):
    parse_array = np.array(parsing)
    parse_person = (parse_array > 0).astype(np.float32)
    parse_background = (parse_array <= 0).astype(np.float32)

    foreground = cut(im, parse_array, parse_person)
    background = cut(im, parse_array, parse_background)

    parts = torch.cat((foreground, background), dim=0)

    return parts


def get_image_tensor(jpg_path, opt):
    image = Image.open(jpg_path).convert('RGB')
    params = get_params(opt, image.size)
    transform = get_transform(opt, params)
    image_tensor = transform(image)

    return image_tensor

def get_theta_from_json(theta_json, tag='aff_tps'):
    theta_list_0 = theta_json[tag]
    theta_len = len(theta_list_0)

    theta_list = [0] * (256 * 256)
    theta_list[:theta_len] = theta_list_0
    theta_tensor = torch.FloatTensor(theta_list)
    theta_tensor = theta_tensor.view(1, 256, 256)

    return theta_tensor


def get_thetas_tensor(theta_json_data, theta_pair_key):
    theta_json = theta_json_data[theta_pair_key]
    theta_aff_tensor = get_theta_from_json(theta_json, tag='aff')
    theta_tps_tensor = get_theta_from_json(theta_json, tag='tps')
    theta_aff_tps_tensor = get_theta_from_json(theta_json, tag='aff_tps')

    return theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor

import pdb
def get_theta_grid_from_json(theta_json, affTnf, tpsInf, tag='aff_tps'):
    theta_list_0 = theta_json[tag]
    theta_tensor = torch.FloatTensor(theta_list_0)
    if tag == 'aff':
        grid = affTnf.get_grid(Variable(theta_tensor.view(-1, 2, 3)))  # (1, 240, 240, 2)
    else:
        theta_tensor = theta_tensor.unsqueeze(0)
        grid = tpsInf.get_grid(Variable(theta_tensor))  # (1, 240, 240, 2)

    b, c, h, w = grid.shape
    theta_len = b * c * h * w

    theta_tmp = grid.view(theta_len).numpy().tolist()
    theta_list = [0] * 2 * (256 * 256)
    theta_list[:theta_len] = theta_tmp
    theta_tensor = torch.FloatTensor(theta_list)
    theta_tensor = theta_tensor.view(2, 256, 256)

    return theta_tensor

def get_theta_grid_from_tensor(theta_tensor, affTnf, tpsTnf, tag):
    if tag == 'aff':
        grid = affTnf.get_grid(theta_tensor.view(-1, 2, 3))  # (1, 240, 240, 2)
    else:
        # theta_tensor = theta_tensor.unsqueeze(0)
        grid = tpsTnf.get_grid(theta_tensor)  # (1, 240, 240, 2)

    b, c, h, w = grid.shape
    theta_len = b * c * h * w
    theta_tmp = grid.view(theta_len)

    import torch.nn.functional as F
    grid_p = F.pad(theta_tmp, (0, 256*256*2-240*240*2))
    theta_tensor = grid_p.view(2,256,256)

    return theta_tensor

def get_theta_affgrid_by_tensor(affTnf, tpsTnf, theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor):
    # pdb.set_trace()
    theta_aff_tensor = get_theta_grid_from_tensor(theta_aff_tensor, affTnf, tpsTnf, tag='aff')
    theta_tps_tensor = get_theta_grid_from_tensor(theta_tps_tensor, affTnf, tpsTnf, tag='tps')
    theta_aff_tps_tensor = get_theta_grid_from_tensor(theta_aff_tps_tensor, affTnf, tpsTnf, tag='aff_tps')
    return theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor

def get_thetas_affgrid_tensor(affTnf, tpsTnf, theta_json_data, theta_pair_key):
    theta_json = theta_json_data[theta_pair_key]
    theta_aff_tensor = get_theta_grid_from_json(theta_json, affTnf, tpsTnf, tag='aff')
    theta_tps_tensor = get_theta_grid_from_json(theta_json, affTnf, tpsTnf, tag='tps')
    theta_aff_tps_tensor = get_theta_grid_from_json(theta_json, affTnf, tpsTnf, tag='aff_tps')

    return theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor

def get_thetas_affgrid_tensor_by_json(affTnf, tpsTnf, theta_json):
    theta_aff_tensor = get_theta_grid_from_json(theta_json, affTnf, tpsTnf, tag='aff')
    theta_tps_tensor = get_theta_grid_from_json(theta_json, affTnf, tpsTnf, tag='tps')
    theta_aff_tps_tensor = get_theta_grid_from_json(theta_json, affTnf, tpsTnf, tag='aff_tps')

    return theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor