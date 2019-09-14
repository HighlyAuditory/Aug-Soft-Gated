from .base_model import BaseModel
import torch
import pdb

from .stage_I_model import Stage_I_Model
from .semantic_align_model import SemanticAlignModel
from models.geo.geo_API import GeoAPI
from .skeleton_model import Skeleton_Model
import models.losses as losses
from models.parsing_loss.parsing_loss import ParsingLoss
import models.networks as networks
from util.image_pool import ImagePool
from .good_order_cood_angle_convert import anglelimbtoxyz2, check_visibility
from . import heatmap_pose
from torch.autograd import Variable
from util.util import parsingim_2_tensor
from data.utils import get_theta_affgrid_by_tensor

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
            # pdb.set_trace()
            result[i, j] = res

    return result

class AugmentModel(BaseModel):
    def name(self):
        return 'AugmentModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.parse_model = Stage_I_Model()
        
        self.parse_model.initialize(opt, "resNet")
        self.parse_model.eval()


        self.main_model = SemanticAlignModel()
        
        self.main_model.initialize(opt, "wapResNet_v3_afftps")

        self.net_SK = Skeleton_Model(opt).cuda()
        self.geo = GeoAPI()

        # cpm_model_path = 'openpose_coco_best.pth.tar'
        # self.cpm_model = heatmap_pose.construct_model(cpm_model_path)

        self.parsing_label_nc = opt.parsing_label_nc
        self.opt = opt

        nb = opt.batchSize
        size = opt.fineSize

        self.mask = torch.ones([nb, 1, 46, 32]).cuda()

        print('---------- Networks initialized -------------')
        
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionFeat = torch.nn.L1Loss()

            # parsing loss for unsupervised pairs
            if not opt.no_Parsing_loss:
                self.criterionParsingLoss = ParsingLoss()
            if not self.opt.no_VGG_loss:
                self.criterionVGG = losses.VGGLoss()
            if not opt.no_TV_loss:
                self.criterionTV = losses.TVLoss()

            self.loss_names = ['D_real', 'D_fake', 'G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_L1', 'G_TV', 'G_Parsing']

            # initialize optimizers
            # optimizer G
            self.optimizer_G = self.main_model.optimizer_G
            # optimizer SK
            self.optimizer_SK = torch.optim.Adam([self.net_SK.alpha], lr=1 , betas=(opt.beta2, 0.999))

    def encode_input(self, inputs, infer=False):
        a_label_tensor = inputs['a_label_tensor']  # 18
        b_label_tensor = inputs['b_label_tensor']  # 18
        # a_label_var = Variable(a_label_tensor.cuda(), volatile=infer)
        # b_label_var = Variable(b_label_tensor.cuda(), volatile=infer)
        a_image_tensor = inputs['a_image_tensor']  # 3
        a_image_tensor = a_image_tensor.type(torch.cuda.FloatTensor)

        a_parsing_tensor = inputs['a_parsing_tensor']  #1
        # a_parsing_var = Variable(a_parsing_tensor)
        b_parsing_tensor = inputs['b_parsing_tensor']
        # b_parsing_var = Variable(b_parsing_tensor)
        size = a_parsing_tensor.size()

        oneHot_size = (size[0], self.opt.parsing_label_nc, size[2], size[3])
        a_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        a_parsing_label = a_parsing_label.scatter_(1, a_parsing_tensor.long().cuda(), 1.0)

        b_parsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        b_parsing_label = b_parsing_label.scatter_(1, b_parsing_tensor.long().cuda(), 1.0)

        a_image_tensor = a_image_tensor.type(torch.cuda.FloatTensor)
        b_label_tensor = b_label_tensor.type(torch.cuda.FloatTensor)
        self.input_tensor_parse = torch.cat([a_parsing_label, b_label_tensor], dim=1)
        # input_var_parse = Variable(input_tensor_parse.cuda(), volatile=infer)

        self.input_tensor_main = torch.cat([a_image_tensor, a_parsing_label, b_parsing_label, b_label_tensor], dim=1)
        # input_var_main = Variable(input_tensor_main.cuda(), volatile=infer)

        return self.input_tensor_parse, self.input_tensor_main

    def get_geo(self, a_parsing, fake_b_parsing):
        theta_aff, theta_tps, theta_aff_tps = self.geo.get_thetas_from_image(a_parsing, fake_b_parsing)
        
        theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor = get_theta_affgrid_by_tensor(self.geo.affTnf, self.geo.tpsTnf, theta_aff, theta_tps, theta_aff_tps)

        # \wenwen{check whether I should be dealt with this}
        theta_aff_tensor = theta_aff_tensor.unsqueeze_(0)
        theta_tps_tensor = theta_tps_tensor.unsqueeze_(0)
        theta_aff_tps_tensor = theta_aff_tps_tensor.unsqueeze_(0)

        theta_tensors = torch.cat([theta_aff_tensor, theta_tps_tensor, theta_aff_tps_tensor], dim=1)

        return theta_tensors, theta_tps_tensor

    def forward_aug(self, inputs, infer):
        a1, a2 = inputs['K1'].cuda().float(), inputs['K2'].cuda().float()
        offset = inputs['F1'].cuda().float() # (b, 3) will be enough
        limbs = inputs['L1'].cuda().float() # (b, 7) will be enough

        self.main_model.train()
        self.net_SK.eval()
        self.parse_model.eval()
        self.encode_input(inputs) 

        aug_angles = self.net_SK(a1, a2)
        aug_pose = anglelimbtoxyz2(offset, aug_angles, limbs)
        
        for i in range(aug_pose.shape[0]):
            aug_pose[i] = check_visibility(aug_pose[i]) # 2d pose

        aug_pose = aug_pose[...,:2]

        self.input_BP_aug = cords_to_map_yx(aug_pose, (256, 256), sigma=0.4).float()
        # paste skeleton2 face
        BP2 = inputs['a_label_tensor'].cuda().float()
        for j in range(4):
            self.input_BP_aug[:, j+14] = BP2[:, j+14]
        self.input_BP_aug[:, 0] = BP2[:, 0]

        parse_input_aug = self.input_tensor_parse
        parse_input_aug[:,20:38] = self.input_BP_aug
        fake_b_parse = self.parse_model.inference(parse_input_aug)
        # fake_b_parse = self.input_tensor_parse

        a_parsing_rgb_tensor = parsingim_2_tensor(parse_input_aug[0,:20], opt=self.opt, parsing_label_nc=self.parsing_label_nc)
        fake_b_parsing_rgb_tensor = parsingim_2_tensor(fake_b_parse[0], opt=self.opt, parsing_label_nc=self.parsing_label_nc)

        theta_tensors,_ = self.get_geo(a_parsing_rgb_tensor, fake_b_parsing_rgb_tensor)

        main_input_aug = self.input_tensor_main
        # main_input_aug[] still need to change
        main_input_final = torch.cat([main_input_aug, fake_b_parse, theta_tensors], dim=1   )
        
        b_prediction = self.main_model.inference(main_input_final)

        reparsing_loss=self.criterionParsingLoss.getSemiParsingLoss(b_prediction, fake_b_parsing_rgb_tensor) * self.opt.lambda_Parsing

        return b_prediction, reparsing_loss    

    def forward_target(self, inputs, infer):
        self.net_SK.train()
        self.main_model.eval()
        self.parse_model.eval()
        fake_b_parse = self.input_tensor_parse
        # fake_b_parse = self.parse_model.inference(self.input_tensor_parse)

        a_parsing_rgb_tensor = parsingim_2_tensor(self.input_tensor_parse[0,:20]  , opt=self.opt, parsing_label_nc=self.parsing_label_nc)
        b_parsing_rgb_tensor = parsingim_2_tensor(fake_b_parse[0], opt=self.opt, parsing_label_nc=self.parsing_label_nc)

        theta_tensors, theta_tps = self.get_geo(a_parsing_rgb_tensor, b_parsing_rgb_tensor)
        main_input_final = torch.cat([self.input_tensor_main, fake_b_parse, theta_tensors], dim=1)
        target_loss, b_prediction = self.main_model.forward(main_input_final, False)

        losses = [ torch.mean(x) for x in target_loss ]
        loss_dict = dict(zip(self.loss_names, losses))

        # loss_D = (loss_dict['D_real'] + loss_dict['D_fake']) * 0.5
        loss_G = loss_dict['G_GAN']  + loss_dict['G_L1'] + loss_dict['G_VGG'] + loss_dict['G_TV'] + loss_dict['G_GAN_Feat'] + loss_dict['G_Parsing']
        
        print(loss_G)
        return b_prediction, b_parsing_rgb_tensor.mean()