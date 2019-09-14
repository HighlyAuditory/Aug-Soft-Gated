import torch
from torch.autograd import Variable
from .base_model import BaseModel
import numpy as np
import os
import pdb

import torch.nn as nn

class Skeleton_Model(BaseModel):
    def name(self):
        return 'Skeleton_Model'

        # input shape (b, 7, 3)

    def __init__(self, opt):
        super(Skeleton_Model, self).__init__()
        BaseModel.initialize(self, opt)
        self.alpha = Variable(torch.randn((1,7,1), device="cuda"), requires_grad=True)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            save_filename = 'alphas_epoch{}.npy'.format(which_epoch)
            alpha_path = os.path.join(self.save_dir, save_filename)
            if os.path.exists(alpha_path):
                val = np.load(alpha_path)
                self.alpha.data = torch.Tensor(val).cuda().data
                print(alpha_path)

    def forward(self, input1, input2):
        self.alpha_clipped = torch.clamp(self.alpha, 0.0, 1.0)
        # self.alpha_clipped = self.alpha
        out = self.alpha_clipped * input1 + (1 - self.alpha_clipped) * input2
        return out

    def save(self, label):
        save_path = os.path.join(self.save_dir, 'alphas_epoch{}.npy'.format(label))
        np.save(save_path, self.alpha.cpu().data)