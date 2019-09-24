#coding=utf-8

import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import losses
from .inter_skeleton_model import InterSkeleton_Model
from . import heatmap_pose
from .good_order_cood_angle_convert import anglelimbtoxyz2, check_visibility
import numpy as np
from PIL import Image
import pdb

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

# will be add to some util file later
def cords_to_map_yx(cords, img_size, sigma=6):
    MISSING_VALUE = -1
    result = torch.zeros([cords.size(0), 18, 256, 256])
    for i, points in enumerate(cords):
        for j in range(14):
            point = points[j]
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            xx, yy = torch.meshgrid([torch.arange(img_size[0], dtype=torch.int32).cuda(), torch.arange(img_size[1],dtype=torch.int32).cuda()])
            xx = xx.float()
            yy = yy.float()
            res = torch.exp(-((yy - point[0]) ** 2 + (xx - point[1] - 40) ** 2) / (2 * sigma ** 2))
            result[i, j] = res

    return result

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs.transpose(0,3,1,2)

class Augment_Stage_I_Model(BaseModel):
    def name(self):
        return 'Augment_Stage_I_Model'

    def initialize(self, opt, which_G):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none':            # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.which_epoch = 100
        self.skeleton_net = InterSkeleton_Model(opt).cuda()

        netG_input_nc = self.opt.parsing_label_nc + 18
        output_nc = self.opt.parsing_label_nc
        self.netG = networks.define_G(netG_input_nc, output_nc, which_G=which_G)
        # Discriminator network
        # will experiment on whether discriminator is needed later
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(netG_input_nc + output_nc, not opt.no_ganFeat_loss)

        if self.isTrain:
            cpm_model_path = 'openpose_coco_b16_best.pth.tar'
            self.cpm_model = heatmap_pose.construct_model(cpm_model_path)
            self.mask = torch.ones([opt.batchSize, 1, 46, 46]).cuda()

        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
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
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionParsingLoss = losses.ParsingCrossEntropyLoss(tensor=self.Tensor)
            self.pose_loss = torch.nn.MSELoss()

            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_L1', 'G_parsing', 'D_real', 'D_fake']

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params_D = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params_D, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer SK
            self.optimizer_SK = torch.optim.Adam([self.skeleton_net.alpha], lr=opt.lr , betas=(opt.beta2, 0.999))
            print("models [%s] was initialized" % (self.name()))

    def label2onhot(self, b_parsing_tensor):
        # print ("label2onehot b_parsing_tensor", b_parsing_tensor.shape)
        size = b_parsing_tensor.size()
        oneHot_size = (size[0], self.opt.parsing_label_nc, size[2], size[3])
        b_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        b_parsing_label = b_parsing_label.scatter_(1, b_parsing_tensor.data.long().cuda(), 1.0)

        return b_parsing_label

    def encode_input(self, data, infer=False):
        a_parsing_tensor = data['a_parsing_tensor']  # 1
        b_parsing_tensor = data['b_parsing_tensor']  # 1
        b_label_tensor = data['b_label_tensor'].cuda()  # 18 BP2

        a_parsing_tensor = self.label2onhot(a_parsing_tensor).cuda()
        b_parsing_tensor = self.label2onhot(b_parsing_tensor).cuda()

        # augment stage input: 3d1, 3d2
        input_tensor = torch.cat((a_parsing_tensor, b_label_tensor), dim=1)

        a1, a2 = data['K1'].cuda().float(), data['K2'].cuda().float()
        offset = data['F1'].cuda().float() # (b, 3) will be enough
        limbs = data['L1'].cuda().float() # (b, 7) will be enough

        return input_tensor, b_parsing_tensor, a_parsing_tensor, b_label_tensor, a1, a2, offset, limbs

    def inference(self, inputs):
        with torch.no_grad():
            
            _, b_parsing_var, a_parsing_var, b_label_var,_,_,_,_ = self.encode_input(inputs, infer=True)
            # self.input_tensor_parse = torch.cat([a_parsing_var, b_label_var], dim=1)
            input_all = torch.cat((a_parsing_var, b_label_var), dim=1)
            fake_b_parsing_var = self.netG.forward(input_all)

        return fake_b_parsing_var

    def forward_augment(self, inputs):
        _, _, a_parsing_tensor, b_label, a1, a2, offset, limbs = self.encode_input(inputs)
        aug_angles = self.skeleton_net(a1, a2)
        # aug_angles = a1
        
        aug_pose = anglelimbtoxyz2(offset, aug_angles, limbs)
        # aug_pose = anglelimbtoxyz2(offset, a1, limbs)

        for i in range(b_label.shape[0]):
            aug_pose[i] = check_visibility(aug_pose[i]) # 2d pose

        aug_pose = aug_pose[...,:2]

        self.input_BP_aug = cords_to_map_yx(aug_pose, (256, 256), sigma=0.4).float().cuda()
        # self.input_BP_aug = BP2
        self.input_BP_res = cords_to_map_yx(aug_pose, (256, 256), sigma=4).cuda().float()

        # paste skeleton2 face
        for j in range(4):
            self.input_BP_aug[:, j+14] = b_label[:, j+14]
        self.input_BP_aug[:, 0] = b_label[:, 0]

        input_aug = torch.cat((a_parsing_tensor, self.input_BP_aug), dim=1)
        fake_aug_parsing = self.netG.forward(input_aug)
        fake_aug_parsing = torch.max(fake_aug_parsing, 1)[1]
        fake_aug_parsing_rgb = torch.Tensor(decode_labels(fake_aug_parsing)).cuda()

        fake_p2_padded = heatmap_pose.preprocess(fake_aug_parsing_rgb, [256, 256])
        heatmap = heatmap_pose.process(self.cpm_model, fake_p2_padded, self.mask)

        self.optimizer_G.zero_grad()
        loss_pose = self.get_parse_loss(self.input_BP_aug, heatmap)
        loss_pose.backward()
        self.optimizer_G.step()

        return loss_pose, fake_aug_parsing

    def get_parse_loss(self, aug_pose, fake_aug_parsing):
        t = Variable(aug_pose, requires_grad=False)
        pl = self.pose_loss(torch.clamp(fake_aug_parsing, min=0, max=1), t)*self.opt.lambda_pose
        return pl

    def forward_target(self, inputs, infer=False):
        input_all, b_parsing_tensor, _, _,_,_,_,_ = self.encode_input(inputs)

        fake_b_parsing_var = self.netG.forward(input_all)

        loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.getZero(), self.getZero(), self.getZero(), self.getZero()
        if not self.opt.no_GAN_loss:
            loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.get_GAN_losses(self.netD, input_all, b_parsing_tensor, fake_b_parsing_var)

        loss_G_L1 = self.getZero()
        if not self.opt.no_L1_loss:
            loss_G_L1 = self.criterionL1(fake_b_parsing_var, b_parsing_tensor) * self.opt.lambda_L1
            
        loss_G_parsing = self.getZero()
        if not self.opt.no_Parsing_loss:
            loss_G_parsing = self.criterionParsingLoss(fake_b_parsing_var, b_parsing_tensor) * self.opt.lambda_Parsing

        return [[ loss_G_GAN, loss_G_GAN_Feat, loss_G_L1, loss_G_parsing, loss_D_real, loss_D_fake], \
                None if not infer else fake_b_parsing_var ]


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

        return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat


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

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_SK.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def getZero(self):
        return Variable(torch.cuda.FloatTensor([0]))

