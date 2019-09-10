#coding=utf-8
import os
import fnmatch
from shutil import copyfile

def get_filename_list(path):
    filename_list = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.png'):
            filename_list.append(filename)

    return filename_list


def get_path_list(path):
    path_list = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.png'):
            path = os.path.join(root, filename)
            path_list.append(path)

    return path_list


# id_00000006_01_3_back_TO_id_00000006_01_1_front__fake_image
# id_00000006_01_1_front_TO_id_00000006_01_2_side__a-b-fake_b.png
# fasionMENJacketsVestsid0000016801_1front.jpg_fasionMENJacketsVestsid0000016801_2side.jpg.png

# id_00001430_01_2_side_TO_id_00001430_01_1_front__a-b-fake_b
# id_00001430/01_2_side.jpg id_00001430/01_1_front.jpg
# id_00000594/04_7_additional.jpg

src_root_path = 'I:/NIPS2018实验结果/Final/gan_L1_feat_vgg_notv_noparsing_afftps_05102228_BK20180729/test_40/images'
dst_root_path = 'I:/NIPS2018实验结果/DSCF_VS_OURS/picked/gan_L1_feat_vgg_notv_noparsing_afftps_05102228_BK20180729/test_40/images'
src_root_path = unicode(src_root_path, 'utf-8')
dst_root_path = unicode(dst_root_path, 'utf-8')

filename_list = get_filename_list(dst_root_path)
for filename in filename_list:
    if filename.find('dscf') == -1:
        a_b_image_filename = filename.replace('fake_image.png', 'a-b-fake_b.png')

        src_path = os.path.join(src_root_path, a_b_image_filename)
        dst_path = os.path.join(dst_root_path, a_b_image_filename)
        copyfile(src_path, dst_path)


# DSCF_root_path = 'I:/NIPS2018实验结果/DSCF_fakeB/generated_images'
# DSCF_root_path = unicode(DSCF_root_path, 'utf-8')
# DSCF_path_list = get_path_list(DSCF_root_path)
#
# for path in DSCF_path_list:
#     for filename in filename_list:
#         a_b_image_filename = filename.replace('fake_image.png', 'a-b-fake_b.png')
#         arr = filename.split('_')
#         key1 = arr[1] + arr[2] + '_' + arr[3]
#         key2 = arr[7] + arr[8] + '_' + arr[9]
#         if path.find(key1) != -1 and path.find(key2) != -1:
#             src_path = path
#             # newfilename = filename + "=="+ os.path.basename(src_path)
#             newfilename = filename + "_dscf_.png"
#             dst_path = os.path.join(dst_root_path, newfilename)
#             print src_path
#             print dst_path
#             copyfile(src_path, dst_path)



