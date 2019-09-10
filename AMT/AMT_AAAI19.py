#coding=utf-8
import os
import fnmatch
import random
import csv
import shutil

def get_randompair_list_VTION(path):
    url_root = 'http://47.100.21.47/image/'
    randompair_list = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*++a_img_tensor.jpg'):
            a_img_path = url_root + filename
            clothes_path = a_img_path.replace('++a_img_tensor.jpg', '++clothes_tensor.jpg')
            pose_path = a_img_path.replace('++a_img_tensor.jpg', '++b_pose_show_tensor.jpg')

            ours_path = a_img_path.replace('++a_img_tensor.jpg', '++refine_fake_b_tensor.jpg')

            viton_filename = os.path.basename(a_img_path)
            viton_dir = a_img_path.replace(viton_filename, '')
            arr = viton_filename.split('=')
            new_viton_filename = arr[1] + '=' + arr[2] + '_' + arr[4] + '=' + arr[5].replace('keypoints.', '') + arr[7] + '=cloth_front_final.png'
            viton_path = os.path.join(viton_dir, new_viton_filename)

            p_list_begin = [a_img_path, clothes_path, pose_path]
            p_list_end = [ours_path, viton_path]
            random.shuffle(p_list_end)
            randompair_list.append(p_list_begin + p_list_end)

    return randompair_list



def write_csv_by_list(randompair_list, path, batch=10, num=1):
    for i in range(num):
        tmp = path
        exe = '_' + str(i) + '.csv'
        path = path.replace('.csv', exe)
        if os.path.exists(path):
            os.remove(path)

        random.shuffle(randompair_list)
        r_list = randompair_list[:batch]
        r_list.insert(0, ['image_person_url','image_clothes_url','image_pose_url', 'image_A_url', 'image_B_url'])
        print r_list
        with open( path, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(r_list)
        path = tmp

if __name__ == "__main__":
    path = 'I:/multi_pose_try_on/v2/AMT/Ours/fig1_1_selected_Final_Test_Coarse/test_100/images'
    path = unicode(path, 'utf-8')
    viton_list = get_randompair_list_VTION(path)

    for l in viton_list:
        print l

    job_1_viton_path = './job_1_viton_20180831.csv'
    job_1_viton_path = unicode(job_1_viton_path, 'utf-8')
    write_csv_by_list(viton_list, job_1_viton_path, batch=100, num=1)








# def output_AMT_file():
#     out_path = 'I:/AMT/deploy_img'
#     path_1 = 'I:/AMT/nip17_dp/test_latest/images'
#     for root, dirnames, filenames in os.walk(path_1):
#         for filename in fnmatch.filter(filenames, '*__NIP17_fake_b_image.jpg'):
#             nips_path = os.path.join(root, filename)
#             fake_path = nips_path.replace('__NIP17_fake_b_image.jpg', '__fake_b_image.jpg')
#             shutil.copy(nips_path, out_path)
#             shutil.copy(fake_path, out_path)