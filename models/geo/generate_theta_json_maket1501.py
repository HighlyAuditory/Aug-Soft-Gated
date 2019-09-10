#coding=utf-8

import os
import sys
import json
from geo_API import GeoAPI

# 从pair文件里面读取，然后一对文件名作为一个key

#pair_file_path = "./datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_mvoneId_0118.txt"



def generate_theta(pair_file_path, root_path, geo):
    theta_json = {}
    file = open(pair_file_path)
    lines = file.readlines()
    count = 0
    for l in lines:
        l = l.strip()
        l = l.split()
        src = l[0].split(os.sep)
        dst = l[1].split(os.sep)
        key = src[-2] + '_' + src[-1] + "=" + dst[-2] + '_' + dst[-1]

        source_image_path, target_image_path = os.path.join(root_path,l[0]), os.path.join(root_path, l[1])
        source_image_path = source_image_path.replace('.jpg', '_vis.png').replace('/128p/', '/128p_parsing/')
        target_image_path = target_image_path.replace('.jpg', '_vis.png').replace('/128p/', '/128p_parsing/')

        theta_aff, theta_tps, theta_aff_tps = geo.get_thetas(source_image_path, target_image_path)
        theta_aff_list = theta_aff.cpu().data.numpy()[0].tolist()
        theta_tps_list = theta_tps.cpu().data.numpy()[0].tolist()
        theta_aff_tps_list = theta_aff_tps.cpu().data.numpy()[0].tolist()

        sub_json = {}
        sub_json['aff'] = theta_aff_list
        sub_json['tps'] = theta_tps_list
        sub_json['aff_tps'] = theta_aff_tps_list
        theta_json[key] = sub_json

        count = count + 1
        if count % 100 == 0:
            print(count)

    return theta_json

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "参数小于1！"
        print sys.argv
        exit()

    index = sys.argv[1]

    geo = GeoAPI()
    root_path = "/home/disk2/donghaoye/datasets/Market-1501/Market-1501-v15.09.15/128p/"
    #pair_file_path = "../../datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_mvoneId_0118.txt"
    #pair_file_path = "../../datasets/Market-1501/pairPATH_paths_20180120.txt"
    pair_file_path = "../../datasets/Market-1501/pairPATH_paths_20180406_{}.txt".format(index)
    out_json_file = "../../datasets/Market-1501/theta_json/maket1501_pair_theta_0406_{}.txt".format(index)
    print pair_file_path
    print out_json_file

    print ("start....")
    theta_json_data = generate_theta(pair_file_path, root_path, geo)

    if os.path.exists(out_json_file):
        os.remove(out_json_file)

    with open(out_json_file, 'w') as outfile:
        json.dump(theta_json_data, outfile)

    # data = json.load(open(path))
    # peoples = data['people']

    print ("done!")

    # CUDA_VISIBLE_DEVICES=1 python generate_theta_json_maket1501.py 1
