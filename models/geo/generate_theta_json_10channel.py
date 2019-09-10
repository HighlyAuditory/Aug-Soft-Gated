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
        l = l.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all_10channel/').split()
        src = l[0].split(os.sep)
        dst = l[1].split(os.sep)
        key = src[-2] + '_' + src[-1] + "=" + dst[-2] + '_' + dst[-1]

        source_image_path, target_image_path = os.path.join(root_path,l[0]), os.path.join(root_path, l[1])
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
    root_path = "/home/disk2/donghaoye/datasets/In-shop_HD/Img/"
    #pair_file_path = "../../datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_mvoneId_0118.txt"
    #pair_file_path = "../../datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_0402.txt"
    pair_file_path = "../../datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_0402_{}.txt".format(index)
    out_json_file = "../../datasets/deepfashion/theta_json/deepfashion_pair_theta_0412_10channel_{}.txt".format(index)
    print pair_file_path
    print out_json_file
    print ("start....")

    theta_json_data = generate_theta(pair_file_path, root_path, geo)

    if os.path.exists(out_json_file):
        os.remove(out_json_file)

    with open(out_json_file, 'w') as outfile:
        json.dump(theta_json_data, outfile)



    print ("done!")


