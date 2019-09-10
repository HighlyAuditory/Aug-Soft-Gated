# -*- coding: utf-8 -*-

import os
from shutil import copyfile

#fasionMENJacketsVestsid0000016802_1front.jpg_fasionMENJacketsVestsid0000016802_4full.jpg.png
#id_00000168_02_2_side_TO_id_00000168_02_4_full__a-b-fake_b.png


#img/MEN/Jackets_Vests/id_00000168/01_1_front.jpg


deformableGAN_results_root = unicode('I:\\NIPS2018实验结果\\deformableGAN的结果\\pose-gan-output\\generated_images_fasion_full_23850','utf8')
# ours_results_root = unicode('I:\\NIPS2018实验结果\\Final_最终结果整理\\gan_L1_feat_vgg_notv_noparsing_afftps_05102228_BK20180729\\test_40\\images','utf8')
ours_results_root = unicode('I:\\NIPS2018实验结果\\Final_最终结果整理\\51111_xintong\\gan_L1_feat_vgg_notv_noparsing_afftps_05102228\\test_40\\images','utf8')

output_defor_root = unicode('I:\\NIPS2018实验结果\\pick_deformableGAN_ours\\deformableGAN_xintong','utf8')
output_ours_root = unicode('I:\\NIPS2018实验结果\\pick_deformableGAN_ours\\softgated_xintong','utf8')

# data_filename = 'I:\\workspace\\semantic_align_gan_v9\\datasets\\deepfashion\\In-shop_AB_HD_p1024\\Anno\\list_landmarks_inshop_filterAll_by_jsonpoint_pairs_0402_shuffle.txt'
data_filename = 'DeepFashion_common_test.txt'
lines = open(data_filename).readlines()
count = 0
print len(lines)
count_set = set()
for line in lines:
    arr = line.split()
    pose_a, pose_b, tag = arr[0], arr[1], arr[2]
    if tag == 'test':
        if count == 1000:
            break

        # deformableGAN
        # fasionMENJacketsVestsid0000016802_1front.jpg
        a_arr = pose_a.split('/')
        b_arr = pose_b.split('/')
        a_tmp = 'fasion' + a_arr[1] + a_arr[2].replace('_', '') + a_arr[3].replace('_', '') +  a_arr[4][:2] + '_' + a_arr[4].replace('_', '')[2:]
        b_tmp = 'fasion' + b_arr[1] + b_arr[2].replace('_', '') + b_arr[3].replace('_', '') +  b_arr[4][:2] + '_' + b_arr[4].replace('_', '')[2:] + '.png'
        filename_deformableGAN = a_tmp + '_' + b_tmp
        s_path_deformableGAN = os.path.join(deformableGAN_results_root, filename_deformableGAN)
        d_path_deformableGAN = os.path.join(output_defor_root, filename_deformableGAN)

        # id_00006177/03_4_full.jpg
        # ours id_00000168_02_2_side_TO_id_00000168_02_4_full__a-b-fake_b.png
        a_tmp = a_arr[-2] + '_' + a_arr[-1].replace('.jpg', '')
        b_tmp = b_arr[-2] + '_' + b_arr[-1].replace('.jpg', '')
        filename_ours = a_tmp + '_TO_' + b_tmp + '__a-b-fake_b.png'
        s_path_ours = os.path.join(ours_results_root, filename_ours)
        d_path_ours = os.path.join(output_ours_root, filename_ours)

        # if os.path.exists(s_path_deformableGAN):
        try:
            copyfile(s_path_deformableGAN, d_path_deformableGAN)
            copyfile(s_path_ours, d_path_ours)
        except:
            # print (s_path_ours)
            33

        count = count + 1
        print count



    else:
        print (22222)
