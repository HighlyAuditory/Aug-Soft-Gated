
def get_path_pairs(pairs_path, phase):
    p = []
    lines = open(pairs_path).readlines()
    for l in lines:
        if l.split()[-1].strip() == phase:
            p.append(l)

    return p

if __name__ == "__main__":
    print 33333333
    train_list = get_path_pairs('../datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_mvoneId_0118.txt', 'train')
    test_list = get_path_pairs('../datasets/deepfashion/In-shop_AB_HD_p1024/Anno/list_landmarks_inshop_filterAll_by_jsonpoint_pairs_mvoneId_0118.txt', 'test')

    print len(train_list)
    print len(test_list)