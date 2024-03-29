#coding=utf-8
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=20, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest models')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained models from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached models')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--num_iterations_per_epoch', type=int, default=1000, help='2000次迭代就完成一个epoch')

        # for discriminators        
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--num_D', type=int, default=3, help='number of discriminators to use')  ### 论文中本来就是采用3个D，3个尺度啊。
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        # for losses
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* no_ganFeat_loss loss')
        self.parser.add_argument('--no_VGG_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_L1_loss', action='store_true', help='if specified, do *not* use L1 loss')
        self.parser.add_argument('--no_GAN_loss', action='store_true', help='if specified, do *not* use GAN loss')
        self.parser.add_argument('--no_TV_loss', action='store_true', help='if specified, do *not* use TV loss')
        self.parser.add_argument('--no_Parsing_loss', action='store_true', help='if specified, do *not* use no_Parsing_loss')

        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for lambda_featloss')
        self.parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for VGG loss')
        self.parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_TV', type=float, default=10e-6, help='weight for TV loss')
        self.parser.add_argument('--lambda_Parsing', type=float, default=0.0001, help='weight for Perceptual loss')

        # self.parser.add_argument('--no_dynamic_policy', action='store_true', help='no_dynamic_policy')
        # self.parser.add_argument('--which_policy', type=str, default='policy1', help='policy1 | policy2 | policy3 | policy4')

        self.isTrain = True
