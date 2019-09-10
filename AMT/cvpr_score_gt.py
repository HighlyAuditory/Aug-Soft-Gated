#coding=utf-8
import csv



if __name__ == '__main__':
    # co_dict = {}
    # path = "I:/AMT/gt/choice.txt"
    # lines = open(path).readlines()
    # for i in range(1, len(lines) - 1):
    #     c = lines[i].split()[0]
    #     hid = lines[i].split()[-1]
    #     print hid
    #     co_dict[hid] = c


    count_nips = 0
    count_ours = 0
    path = "I:/实验室结果/AMT/cvpr2018_Batch_3236353_batch_results.csv"
    path = unicode(path, 'utf-8')
    for row in csv.DictReader(open(path)):
        a = row['Input.image_A_url']
        b = row['Input.image_B_url']
        choice = row['Answer.choice']
        # hid = row['HITId']
        # choice = co_dict[hid]
        if choice == 'optionA':
            if a.endswith('__CVPR18_fake_b_image.png'):
                print a
                print choice
                count_nips = count_nips + 1
            else:
                count_ours = count_ours + 1
        elif choice == 'optionB':
            if b.endswith('__CVPR18_fake_b_image.png'):
                count_nips = count_nips + 1
            else:
                count_ours = count_ours + 1


    print(count_nips, count_ours)

    print float(count_ours*1.0/(count_ours + count_nips)*1.0)