#coding=utf-8

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import sys
sys.path.append('/home/disk2/donghaoye/ACMMM/semantic_align_gan_v9')

import torch
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from torch.autograd import Variable
from util.util import tensor2im, parsingim_2_tensor


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


# no gan
opt.name = "nogan_L1_feat_vgg_notv_noparsing_afftps_051220"
opt.which_G = "wapResNet_v3_afftps"
opt.stage = 2
model_2 = create_model(opt)

# no PH
opt.name = "gan_L1_nofeat_vgg_notv_noparsing_afftps_051116"
opt.which_G = "wapResNet_v3_afftps"
opt.stage = 2
model_3 = create_model(opt)

# no VGG
opt.name = "gan_L1_feat_novgg_notv_noparsing_afftps_051116"
opt.which_G = "wapResNet_v3_afftps"
opt.stage = 2
model_4 = create_model(opt)

# no L1
opt.name = "gan_noL1_feat_vgg_notv_noparsing_afftps_051220"
opt.which_G = "wapResNet_v3_afftps"
opt.stage = 2
model_5 = create_model(opt)

# no warping-block
opt.name = "gan_L1_feat_vgg_notv_noparsing_resnet_05102228"
opt.which_G = "resNet"
opt.stage = 2
model_6 = create_model(opt)

# no stage I
opt.name = "w_o_stageI_gan_L1_feat_vgg_notv_noparsing_afftps_05102320"
opt.which_G = "resNet"
opt.stage = 3
model_7 = create_model(opt)

# Full
opt.name = "gan_L1_feat_vgg_notv_noparsing_afftps_05102228"
opt.which_G = "wapResNet_v3_afftps"
opt.stage = 2
opt.which_epoch=20
model_1 = create_model(opt)

visualizer = Visualizer(opt)
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)


def main():
    web_dir = os.path.join(opt.results_dir, "multi_models_full_woWB_woStageI", '%s_%s' % (opt.phase, opt.which_epoch))
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
        a_jpg_path = data['a_jpg_path']
        b_jpg_path = data['b_jpg_path']

        input_tensor = torch.cat([a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor, b_parsing_tensor, \
                                  theta_aff, theta_tps, theta_aff_tps], dim=1)
        input_var = Variable(input_tensor.type(torch.cuda.FloatTensor))

        fake_b_1 = model_1.inference(input_var)
        fake_b_2 = model_2.inference(input_var)
        fake_b_3 = model_3.inference(input_var)
        fake_b_4 = model_4.inference(input_var)
        fake_b_5 = model_5.inference(input_var)
        fake_b_6 = model_6.inference(input_var)
        fake_b_7 = model_7.inference(input_var)

        # show_image_tensor_1 = torch.cat((a_image_tensor, b_label_show_tensor, b_image_tensor), dim=3)
        # show_image_tensor_2 = torch.cat((a_parsing_rgb_tensor, b_parsing_rgb_tensor, fake_b.data.cpu()), dim=3)
        # show_image_tensor = torch.cat((show_image_tensor_1, show_image_tensor_2), dim=2)
        # test_list = [('a | b | fake_b', tensor2im(show_image_tensor[0]))]

        a_parsing_rgb_tensor = parsingim_2_tensor(a_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)
        b_parsing_rgb_tensor = parsingim_2_tensor(b_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)

        # show_image_tensor_1 = torch.cat((a_image_tensor, b_label_show_tensor, b_image_tensor), dim=3)
        # show_image_tensor_2 = torch.cat((a_parsing_rgb_tensor, b_parsing_rgb_tensor, fake_b.data[0:1, :, :, :].cpu()),dim=3)
        # show_image_tensor = torch.cat((show_image_tensor_1[0:1, :, :, :], show_image_tensor_2), dim=2)
        test_list = [#('a-b-fake_b', tensor2im(show_image_tensor[0])),
                     ('a_image', util.tensor2im(a_image_tensor[0])),
                     ('b_keypoints', util.tensor2im(b_label_show_tensor[0])),
                     ('b_image', util.tensor2im(b_image_tensor[0])),
                     ('full_fake', util.tensor2im(fake_b_1.data[0])),
                     ('noGAN_fake', util.tensor2im(fake_b_2.data[0])),
                     ('noPH_fake', util.tensor2im(fake_b_3.data[0])),
                     ('noVGG_fake', util.tensor2im(fake_b_4.data[0])),
                     ('noL1_fake', util.tensor2im(fake_b_5.data[0])),
                     ('woWB_fake', util.tensor2im(fake_b_6.data[0])),
                     ('woStageI_fake', util.tensor2im(fake_b_7.data[0]))
                     ]

        ### save image
        visuals = OrderedDict(test_list)
        visualizer.save_images(webpage, visuals, a_jpg_path[0], b_jpg_path[0])

        if i % 100 ==0:
            print('[%s]process image... %s' % (i, a_jpg_path[0]))

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



