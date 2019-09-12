#coding=utf-8
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import losses
from .parsing_loss.parsing_loss import ParsingLoss

class SemanticAlignModel(BaseModel):
    def name(self):
        return 'SemanticAlignModel'

    def initialize(self, opt, which_G):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.which_epoch = 40

        # Generator network
        netG_input_nc = 3 + 18 + 2*opt.parsing_label_nc   ### a_parsing_label and b_parsing_label
        output_nc = opt.output_nc
        self.netG = networks.define_G(netG_input_nc, output_nc, which_G=which_G)

        # Discriminator network
        if self.isTrain:
            #use_sigmoid = opt.no_lsgan   ### 为啥用lsGAN就要用sigmoid？
            self.netD = networks.define_D(netG_input_nc + 3, not opt.no_ganFeat_loss, num_D=self.opt.num_D)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)

        if self.isTrain:
            networks.print_network(self.netD)
            print('----------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            print ("pretrained_path = {}, epoch {}".format(pretrained_path, self.which_epoch))

            self.load_network(self.netG, 'G', self.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', self.which_epoch, pretrained_path)


        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = losses.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionFeat = torch.nn.L1Loss()

            if not opt.no_Parsing_loss:
                self.criterionParsingLoss = ParsingLoss()
            if not self.opt.no_VGG_loss:
                self.criterionVGG = losses.VGGLoss()
            if not opt.no_TV_loss:
                self.criterionTV = losses.TVLoss()

            self.loss_names = [ 'D_real', 'D_fake', 'G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_L1', 'G_TV', 'G_Parsing']

            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params_D = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params_D, lr=opt.lr, betas=(opt.beta1, 0.999))

            print("models [%s] was created" % (self.name()))

    def encode_input(self, inputs, infer=False):
        a_image_tensor = inputs.data[:, 0:3, :, :]      #3
        b_image_tensor = inputs.data[:, 3:6, :, :]      #3
        b_label_tensor = inputs.data[:, 6:24, :, :]     #18
        a_parsing_tensor = inputs.data[:, 24:25, :, :]  #1
        b_parsing_tensor = inputs.data[:, 25:26, :, :]  #1
        theta_aff = inputs.data[:, 26:28, :, :]         #2
        theta_tps = inputs.data[:, 28:30, :, :]         #2
        theta_aff_tps = inputs.data[:, 30:32, :, :]     #2


        b, c, h, w = theta_aff_tps.shape
        theta_aff_tensor = theta_aff.view(-1, 2 * h * w)
        theta_tps_tensor = theta_tps.view(-1, 2 * h * w)
        theta_aff_tps_tensor = theta_aff_tps.view(-1, 2 * h * w)

        # one-hot vector for a_parsing_label and b_parsing_label map
        a_parsing_var = Variable(a_parsing_tensor)
        size = a_parsing_var.size()
        oneHot_size = (size[0], self.opt.parsing_label_nc, size[2], size[3])
        a_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        a_parsing_label = a_parsing_label.scatter_(1, a_parsing_var.data.long().cuda(), 1.0)

        b_parsing_var = Variable(b_parsing_tensor)
        b_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        b_parsing_label = b_parsing_label.scatter_(1, b_parsing_var.data.long().cuda(), 1.0)

        a_image_tensor = a_image_tensor.type(torch.cuda.FloatTensor)
        b_label_tensor = b_label_tensor.type(torch.cuda.FloatTensor)
        input_tensor = torch.cat((a_image_tensor, a_parsing_label, b_parsing_label, b_label_tensor), dim=1)
        input_var = Variable(input_tensor.cuda(), volatile=infer)

        # real target images for training
        if b_image_tensor is not None:
            real_image_var = Variable(b_image_tensor.cuda())

        theta_aff_var = Variable(theta_aff_tensor.cuda())
        theta_tps_var = Variable(theta_tps_tensor.cuda())
        theta_aff_tps_var = Variable(theta_aff_tps_tensor.cuda())

        return input_var, real_image_var, theta_aff_var, theta_tps_var, theta_aff_tps_var


    def forward(self, inputs, infer):
        input_var, real_image, theta_aff_var, theta_tps_var, theta_aff_tps_var = self.encode_input(inputs)
        fake_image = self.netG.forward(input_var, theta_aff_var, theta_tps_var, theta_aff_tps_var)

        loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.getZero(), self.getZero(), self.getZero(), self.getZero()
        if not self.opt.no_GAN_loss:
            loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.get_GAN_losses(self.netD, input_var, real_image, fake_image)

        loss_G_VGG = self.getZero()
        if not self.opt.no_VGG_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_VGG

        loss_G_L1 = self.getZero()
        if not self.opt.no_L1_loss:
            loss_G_L1 = self.criterionL1(fake_image, real_image) * self.opt.lambda_L1

        loss_G_TV = self.getZero()
        if not self.opt.no_TV_loss:
            loss_G_TV = self.criterionTV(fake_image) * self.opt.lambda_TV

        loss_G_parsing = self.getZero()
        if not self.opt.no_Parsing_loss:
            loss_G_parsing = self.criterionParsingLoss(fake_image, real_image) * self.opt.lambda_Parsing

        return [[ loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_L1, loss_G_TV, loss_G_parsing], \
                None if not infer else fake_image ]


    def inference(self, inputs):
        with torch.no_grad():
            input_var, _, theta_aff_var, theta_tps_var, theta_aff_tps_var = self.encode_input(inputs, infer=True)
            fake_image = self.netG.forward(input_var, theta_aff_var, theta_tps_var, theta_aff_tps_var)

        return fake_image

    def get_GAN_losses(self, netD, input_label, real_image, fake_image):
        loss_D_fake, loss_D_real, loss_G_GAN, loss_G_GAN_Feat = self.getZero(), self.getZero(), self.getZero(), self.getZero()

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(netD, input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(netD, input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # G GAN loss (Fake Passability Loss)
        pred_fake = netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # discriminator feature matching
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat
                    # print loss_G_GAN_Feat
                    # print "================="

        return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat

    def getZero(self):
        return Variable(torch.cuda.FloatTensor([0]))

    def discriminate(self, netD, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return netD.forward(fake_query)
        else:
            return netD.forward(input_concat)

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        if not self.opt.no_GAN_loss:
            self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    # def update_fixed_params(self):
    #     # after fixing the global generator for a number of iterations, also start finetuning it
    #     params = list(self.netG.parameters())
    #     self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
    #     print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
