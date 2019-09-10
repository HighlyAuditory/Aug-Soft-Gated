#coding=utf-8
import os
import fnmatch
import random
import csv
import shutil

def get_randompair_list_GT(path):
    url_root = 'http://47.100.21.47/image/'
    randompair_list = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*real_b_image.png'):
            a_img_path = url_root + filename
            b_imge_path = a_img_path.replace('real_b_image.png', 'fake_b_image.png')

            p_list = [a_img_path, b_imge_path]
            random.shuffle(p_list)
            randompair_list.append(p_list)

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
        r_list.insert(0, ['image_A_url', 'image_B_url'])
        print r_list
        with open( path, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(r_list)
        path = tmp

if __name__ == "__main__":
    path = 'I:/NIPS2018实验结果/AMT/nip17_dp/test_45/images'
    path = unicode(path, 'utf-8')
    viton_list = get_randompair_list_GT(path)

    for l in viton_list:
        print l

    job_1_gt_path = './dp_vs_GT_201801025.csv'
    job_1_gt_path = unicode(job_1_gt_path, 'utf-8')
    write_csv_by_list(viton_list, job_1_gt_path, batch=100, num=1)


