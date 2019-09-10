#coding=utf-8
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
# from models.parsing_loss.res_net import ResGenerator
from res_net import ResGenerator

class ParsingCrossEntropyLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(ParsingCrossEntropyLoss, self).__init__()
        self.Tensor = tensor
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, input, target):
        # input
        input = input.transpose(0, 1)
        c = input.size()[0]
        n = input.size()[1] * input.size()[2] * input.size()[3]
        input = input.contiguous().view(c, n)
        input = input.transpose(0, 1)

        # target
        [_, argmax] = target.max(dim=1)
        target = argmax.view(n)

        return self.loss(input, target)




def load_network(network, save_dir, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    network.load_state_dict(torch.load(save_path))

class ParsingLoss(nn.Module):
    def __init__(self):
        super(ParsingLoss, self).__init__()
        save_dir = './models/parsing_loss/ckpt'
        which_epoch = '30'
        #self.criterionL1 = nn.L1Loss()
        self.criterionParsingLoss = ParsingCrossEntropyLoss(tensor=torch.cuda.FloatTensor)

        self.model = ResGenerator(3, 20)
        self.model = self.model.cuda()
        self.model.eval()
        load_network(self.model, save_dir, 'G', which_epoch)

    def getParsingLoss(self, img_1, img_2):
        img_1_parsing = self.model.forward(img_1)
        img_2_parsing = self.model.forward(img_2)
        loss_G_parsing = self.criterionParsingLoss(img_1_parsing, img_2_parsing)
        #loss_G_parsing = self.criterionL1(Variable(img_1_parsing.data), Variable(img_2_parsing.data))
        return loss_G_parsing

    def forward(self, img_1, img_2):
        return self.getParsingLoss(img_1, img_2)

