### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        #self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        #self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached models')
        self.parser.add_argument('--which_img', type=str, default='paper_img', help='which epoch to load? set to latest to use latest cached models')
        #self.parser.add_argument('--how_many', type=int, default=12800, help='how many test images to run')
        self.parser.add_argument('--how_many', type=int, default=12821, help='how many test images to run')
        self.parser.add_argument('--pose_file_path', type=str, default='./rebuttal/pose_sequence_20180730.txt', help='how many test images to run')

        self.isTrain = False