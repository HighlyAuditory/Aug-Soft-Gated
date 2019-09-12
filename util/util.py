#coding=utf-8
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from __future__ import print_function
import torch
import numpy as np
from PIL import Image, ImageChops, ImageOps
import os
import cv2 as cv
import torchvision.transforms as transforms
import scipy.io as sio
from data.base_dataset import get_transform, get_params
import pdb
#CMAP = sio.loadmat('human_colormap.mat')['colormap']

CMAP = sio.loadmat('colormap.mat')['colormap']
#CMAP = sio.loadmat('colormap2.mat')['colormap']
CMAP = (CMAP * 256).astype(np.uint8)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def label_2_onhot(b_parsing_tensor, parsing_label_nc=20):
    size = b_parsing_tensor.size()
    oneHot_size = (parsing_label_nc, size[1], size[2])
    b_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    b_parsing_label = b_parsing_label.scatter_(0, b_parsing_tensor.long().cuda(), 1.0)

    return b_parsing_label

def parsing2im(parsing, imtype=np.uint8):
    # parsing_numpy = parsing.cpu().float().numpy()
    CMAP_tensor = torch.Tensor(CMAP)
    image_index = torch.argmax(parsing, dim=0)
    parsing_numpy = torch.zeros((image_index.shape[0], image_index.shape[1], 3))
    for h in range(image_index.shape[0]):
        for w in range(image_index.shape[1]):
            parsing_numpy[h, w, :] = CMAP_tensor[image_index[h, w]]

    return parsing_numpy

def parsingim_2_tensor(parsing_label, opt, parsing_label_nc=20):
    # one_hot = label_2_onhot(parsing_label, parsing_label_nc=parsing_label_nc)
    # label_rgb = parsing2im(one_hot)
    label_rgb = parsing_label[:3]
    # pdb.set_trace()
    # label_rgb_tensor = get_image_tensor_by_im(label_rgb, opt=opt)
    # \wenwens{still need to normalize etc}
    # label_rgb_tensor = label_rgb.transpose(1,2).transpose(0,1)
    label_rgb_tensor = label_rgb.unsqueeze_(0).cuda()

    return label_rgb_tensor


def get_image_tensor_by_im(image, opt):
    #image = Image.open(jpg_path).convert('RGB')
    image = Image.fromarray(image)
    # pdb.set_trace()
    params = get_params(opt, image.size()[:2])
    transform = get_transform(opt, params)
    image_tensor = transform(image)

    return image_tensor

# label_colours = [(0,0,0)
#                 # 0=Background
#                 ,(1,1,1),(2,2,2),(0,85,0),(170,0,51),(255,85,0)
#                 # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
#                 ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
#                 # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
#                 ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
#                 # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
#                 ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
#                 # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

def parsing_2_onechannel(parsing, imtype=np.uint8):
    parsing_numpy = parsing.cpu().float().numpy()
    image_index = np.argmax(parsing_numpy, axis=0)
    parsing_numpy = np.zeros((image_index.shape[0], image_index.shape[1], 3))
    for h in range(image_index.shape[0]):
        for w in range(image_index.shape[1]):
            parsing_numpy[h, w, :] = image_index[h, w]

    return parsing_numpy.astype(imtype)[:, :, 0:1]




def tensor2label(output, n_label, imtype=np.uint8):
    output = output.cpu().float()    
    if output.size()[0] > 1:
        output = output.max(0, keepdim=True)[1]
    output = Colorize(n_label)(output)
    output = np.transpose(output.numpy(), (1, 2, 0))
    return output.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def CV2PIL(im):
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    return im

def PIL2CV(im):
    im = np.array(im)
    # Convert RGB to BGR, CV和PIL无法是两者的RGB顺序不一样, 对于单通道来说，都一样。
    im = im[:, :, ::-1].copy()
    return im

def resize(image, size=(80,80), pad=False):
    #image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size
    if pad:
        thumb = image.crop( (0, 0, size[0], size[1]) )

        offset_x = max( (size[0] - image_size[0]) / 2, 0 )
        offset_y = max( (size[1] - image_size[1]) / 2, 0 )

        thumb = ImageChops.offset(thumb, offset_x, offset_y)
    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))
    return thumb

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = 0
            g = 0
            b = 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def get_a_part(real_img, fake_img, size=(256, 256)):
    head_n = np.array(real_img)
    #head_n = real_img
    head_tmp = (head_n > 0).astype('int64')
    head_ax = np.nonzero(head_tmp)
    if len(head_ax) > 0:
        x0, y0, x1, y1 = min(head_ax[2]) if len(head_ax[2]) > 0 else 0, \
                         min(head_ax[1]) if len(head_ax[1]) > 0 else 0, \
                         max(head_ax[2]) if len(head_ax[2]) > 0 else 0, \
                         max(head_ax[1]) if len(head_ax[1]) > 0 else 0
        box = (x0, y0, x1, y1)
    else:
        box = (0, 0, 256, 256)

    # 出现没有头部的情况，给定一个黑框
    if sum(box) < 128 * 2:
        box = (0, 0, 256, 256)

    real_im = Image.fromarray(np.uint8(real_img))
    fake_im = Image.fromarray(np.uint8(fake_img))

    real_im = real_im.crop(box)
    real_im = resize(real_im, size=size, pad=True)

    fake_im = fake_im.crop(box)
    fake_im = resize(fake_im, size=size, pad=True)

    return real_im, fake_im



def get_a_part_for_lightCNN(real_img, fake_img, size=(128, 128)):
    head_n = np.array(real_img)
    #head_n = real_img
    head_tmp = (head_n > 0).astype('int64')
    head_ax = np.nonzero(head_tmp)
    if len(head_ax) > 0:
        x0, y0, x1, y1 = min(head_ax[2]) if len(head_ax[2]) > 0 else 0, \
                         min(head_ax[1]) if len(head_ax[1]) > 0 else 0, \
                         max(head_ax[2]) if len(head_ax[2]) > 0 else 0, \
                         max(head_ax[1]) if len(head_ax[1]) > 0 else 0
        box = (x0, y0, x1, y1)
    else:
        box = (0, 0, 128, 128)

    # 出现没有头部的情况，给定一个黑框
    if sum(box) < 128 * 2:
        box = (0, 0, 128, 128)

    real_im = Image.fromarray(np.uint8(real_img))
    fake_im = Image.fromarray(np.uint8(fake_img))

    real_im = real_im.crop(box)
    real_im = resize(real_im, size=size, pad=True)

    fake_im = fake_im.crop(box)
    fake_im = resize(fake_im, size=size, pad=True)

    return real_im, fake_im

def im2LightCNN_input(image, volatile=False):
    input_tmp = torch.zeros(1, 1, 128, 128)
    transform = transforms.Compose([transforms.ToTensor()])
    fake_tmp = np.reshape(image, (128, 128, 1))
    fake_tmp = transform(fake_tmp)
    input_tmp[0, :, :, :] = fake_tmp
    input_tmp = input_tmp.cuda()

    return torch.autograd.Variable(input_tmp, volatile=volatile)

def get_display_image(part_list):
    foreground = tensor2im(part_list[0])
    for k in range(1, len(part_list) - 1):
        foreground += tensor2im(part_list[k])
    full = foreground + tensor2im(part_list[-1])

    row_1, row_2, row_3 = [], [], []
    for i in range(len(part_list)):
        if i < 4:
            row_1.append(tensor2im(part_list[i]))
        elif i >= 4 and i < 8:
            row_2.append(tensor2im(part_list[i]))
        elif i >= 8:
            row_3.append(tensor2im(part_list[i]))

    row_3.append(foreground)
    row_3.append(full)

    row_1_img, row_2_img, row_3_img = [], [], []
    if len(row_1) > 0:
        row_1_img = np.concatenate(row_1, axis=1)
    if len(row_2) > 0:
        row_2_img = np.concatenate(row_2, axis=1)
    if len(row_3) > 0:
        row_3_img = np.concatenate(row_3, axis=1)

    rows = []
    if len(row_1_img) > 0:
        rows.append(row_1_img)
    if len(row_2_img) > 0:
        rows.append(row_2_img)
    if len(row_3_img) > 0:
        rows.append(row_3_img)

    rows_img = np.concatenate(rows, axis=0)

    return rows_img


def one_hot(batch_size, label, dim):
    """Convert label indices to one-hot vector"""
    # batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    #out[np.arange(batch_size), label*1L] = 1
    for i in range(batch_size):
        out[i, label*1] = 1

    return out


