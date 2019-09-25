import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

import torch.nn as nn

class Skeleton_Model(BaseModel):
    def name(self):
        return 'Skeleton_Model'
        # input shape (b, 3, 7)

    def __init__(self, opt):
        super(Skeleton_Model, self).__init__()
        self.batchSize = opt.batchSize
        BaseModel.initialize(self, opt)
        self.main = nn.Sequential(
            nn.Linear(14, 7)
            # nn.Sigmoid()
        )

    def forward(self, input):
    	# input angle and joints
        # input = input.view(self.batchSize, -1)
        out = self.main(input)
        return out.view(-1, 3, 7)