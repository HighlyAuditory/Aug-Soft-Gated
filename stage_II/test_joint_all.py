#coding=utf-8

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys
sys.path.append('/home/disk2/donghaoye/ACMMM/semantic_align_gan_v3')

import torch
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from torch.autograd import Variable

from PIL import Image
from data.base_dataset import get_transform, get_params
from util.util import tensor2im, parsingim_2_tensor

from val import get_test_result
#from eval.test_is_2 import get_IS
import fnmatch


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)


def main():
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        a_image_tensor = data['a_image_tensor']         # 3
        b_image_tensor = data['b_image_tensor']         # 3
        b_label_tensor = data['b_label_tensor']         # 18
        a_parsing_tensor = data['a_parsing_tensor']     # 1
        b_parsing_tensor = data['b_parsing_tensor']     # 1
        b_label_show_tensor = data['b_label_show_tensor']
        theta_aff = data['theta_aff_tensor']            # 2
        theta_tps = data['theta_tps_tensor']            # 2
        theta_aff_tps = data['theta_aff_tps_tensor']    # 2
        policy_binary = data['policy_binary']           # 1
        a_jpg_path = data['a_jpg_path']
        b_jpg_path = data['b_jpg_path']

        input_tensor = torch.cat([a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor, b_parsing_tensor, \
                                  theta_aff, theta_tps, theta_aff_tps, policy_binary], dim=1)
        input_var = Variable(input_tensor.type(torch.cuda.FloatTensor))
        model.eval()
        fake_b = model.inference(input_var)

        # test_list = [('b_label_show', util.tensor2im(b_label_show_tensor[0])),
        #               ('a_image', util.tensor2im(a_image_tensor[0])),
        #              ('fake_b_parsing', util.tensor2im(
        #                  util.parsingim_2_tensor(b_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)[0])),
        #               ('fake_image', util.tensor2im(fake_b.data[0])),
        #               ('b_image', util.tensor2im(b_image_tensor[0]))]

        a_parsing_rgb_tensor = parsingim_2_tensor(a_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)
        b_parsing_rgb_tensor = parsingim_2_tensor(b_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)

        show_image_tensor_1 = torch.cat((a_image_tensor, b_label_show_tensor, b_image_tensor), dim=3)
        show_image_tensor_2 = torch.cat((a_parsing_rgb_tensor, b_parsing_rgb_tensor, fake_b.data[0:1, :, :, :].cpu()),
                                        dim=3)
        show_image_tensor = torch.cat((show_image_tensor_1[0:1, :, :, :], show_image_tensor_2), dim=2)
        test_list = [('a-b-fake_b', tensor2im(show_image_tensor[0])),
                     ('fake_image', util.tensor2im(fake_b.data[0])),
                     ('b_image', util.tensor2im(b_image_tensor[0]))]

        ### save image
        visuals = OrderedDict(test_list)
        visualizer.save_images(webpage, visuals, a_jpg_path[0], b_jpg_path[0])

        print('[%s]process image %s' % (i, a_jpg_path[0]))
        ### 从零开始为啥只有12779张？本来12800的！难道有11pair是重复的？检查pair文件。。
        ### 奇怪哦！难道要12800 + 21

    webpage.save()

    image_dir = webpage.get_image_dir()
    print image_dir
    # /results/mGPU_nofusion_noD2D3_lightCNN_tv_corr_sia_block3_bz6_0115/test_latest/images



if __name__ == "__main__":
    flag = opt.which_img

    if flag == 'all':
        main()

    else:
        print ('wrong which_img: ' + flag)



