#coding=utf-8

import json
import torch
from PIL import Image, ImageDraw

point_index_list = [[16, 14, 0, 15, 17], [8, 2, 1, 5, 11, 0], [8, 2, 1, 5, 11, 0], [10, 9, 8, 11, 12, 13], \
                    [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]

line_index_list = [[(16, 14), (14, 0), (0, 15), (15, 17)], [(8, 2), (2, 1), (1, 5), (5, 11), (1, 0)], \
                   [(8, 2), (2, 1), (1, 5), (5, 11), (1, 0)], [(10, 9), (9, 8), (8, 11), (11, 12), (12, 13)], \
                   [(2, 3), (3, 4)], [(5, 6), (6, 7)], [(8, 9), (9, 10)], [(11, 12), (12, 13)]]

def get_points(path):
    print(path)
    data = json.load(open(path))
    peoples = data['people']
    point_list = peoples[0]['pose_keypoints'] if len(peoples) > 0 else [0] * 54

    return point_list

def draw_points(point_list, size=(256, 256), r= 4):
    img_blank = Image.new('RGB', size)
    img_draw = ImageDraw.Draw(img_blank)
    for i in xrange(0, len(point_list), 3):
        x,  y = point_list[i], point_list[i+1]
        if x == 0 and y == 0:
            continue
        img_draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(0, 0, 0))

    return img_blank

def draw_18chnl_points(point_list, transform, num_chnl=18, size=(256, 256), r = 4):
    pose_18map = torch.zeros(num_chnl, size[0], size[1])
    map_index = 0
    for i in xrange(0, len(point_list), 3):
        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        x, y = point_list[i], point_list[i + 1]
        if x != 0 and y != 0:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(1, 1, 1), outline=(0, 0, 0))

        img_tensor = transform(img)
        pose_18map[map_index] = img_tensor[0:1]
        map_index = map_index + 1

    return pose_18map

def draw_part_points(point_list, transform, num_chnl=8, size=(256, 256), r=4):
    part_8map = torch.zeros(num_chnl, size[0], size[1])

    for k in range(len(point_index_list)):
        points = point_index_list[k]
        lines = line_index_list[k]

        img = Image.new('RGB', size)
        draw = ImageDraw.Draw(img)
        for i in points:
            x = point_list[i * 3]
            y = point_list[i * 3 + 1]
            if x != 0 and y != 0:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(1, 1, 1), outline=(0, 0, 0))
                # draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(255, 255, 255))

        for line in lines:             # 画线
            (a, b) = line
            a_x, a_y = point_list[a * 3], point_list[a * 3 + 1]
            b_x, b_y = point_list[b * 3], point_list[b * 3 + 1]
            xy = (a_x, a_y, b_x, b_y)
            if a_x != 0 and a_x != 0 and b_x != 0 and b_x != 0:
                draw.line(xy, fill=(1, 1, 1), width=2)
                # draw.line(xy, fill=(255, 255, 255), width=2)

        img_tensor = transform(img)
        part_8map[k] = img_tensor[0:1].type(torch.FloatTensor)

    return part_8map