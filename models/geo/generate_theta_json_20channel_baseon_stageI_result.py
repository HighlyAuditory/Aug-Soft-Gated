#coding=utf-8

import os
import json
from geo_API import GeoAPI


def generate_theta(pair_file_path, root_path, stage_I_result_dir, geo):
    theta_json = {}
    file = open(pair_file_path)
    lines = file.readlines()
    count = 0
    for l in lines:
        l = l.strip()
        l = l.replace('.jpg', '_vis.png').replace('img/', 'img_parsing_all/').split()
        src = l[0].split(os.sep)
        dst = l[1].split(os.sep)
        key = src[-2] + '_' + src[-1] + "=" + dst[-2] + '_' + dst[-1]
        b_parsing_label_filename = key.replace('=', '_TO_')
        b_parsing_label_filename = b_parsing_label_filename.replace('_vis.png', '') + '__fake_b_parsing_RGB.png'
        b_parsing_path = os.path.join(stage_I_result_dir, b_parsing_label_filename)

        source_image_path = os.path.join(root_path, l[0])
        target_image_path = b_parsing_path

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
    geo = GeoAPI()
    root_path = "/home/wenwens/Desktop/Img/"
    stage_I_result_dir = '/home/wenwens/Downloads/stage_I_all_result_train_test_100/images'
    pair_file_path = "./datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_0402.txt"
    theta_json_data = generate_theta(pair_file_path, root_path, stage_I_result_dir, geo)

    out_json_file = "./datasets/deepfashion/theta_json/deepfashion_pair_theta_0502_20channel_all_baseon_stageI_result.txt"
    if os.path.exists(out_json_file):
        os.remove(out_json_file)

    with open(out_json_file, 'w') as outfile:
        json.dump(theta_json_data, outfile)

    print ("done!")