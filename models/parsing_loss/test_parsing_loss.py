#coding=utf-8
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# from lip_dataset import resize_image, parsing2Image
import torch.nn.functional as F
from parsing_loss import ParsingCrossEntropyLoss
from res_net import ResGenerator
#import cv2 as cv

which_epoch = '30'
save_dir = 'ckpt/'
Tensor = torch.cuda.FloatTensor

#按照指定图像大小调整尺寸
def resize_image(image, height, width, BLACK):
     top, bottom, left, right = (0, 0, 0, 0)

     #获取图像尺寸
     h, w= image.shape[0], image.shape[1]

     #对于长宽不相等的图片，找到最长的一边
     longest_edge = max(h, w)

     #计算短边需要增加多上像素宽度使其与长边等长
     if h < longest_edge:
         dh = longest_edge - h
         top = dh // 2
         bottom = dh - top
     elif w < longest_edge:
         dw = longest_edge - w
         left = dw // 2
         right = dw - left
     else:
         pass

     #RGB颜色
     #BLACK = [0, 0, 0]
     # BLACK = [255, 255, 255]

     #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
     constant = cv.copyMakeBorder(image, top , bottom, left, right, cv.BORDER_CONSTANT, value = BLACK)

     #调整图像大小并返回
     return cv.resize(constant, (height, width), cv.INTER_NEAREST)

def load_network(network, save_dir, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    network.load_state_dict(torch.load(save_path))


def trans2Tensor(img):
    img = np.array(img)
    #img = resize_image(img, 256, 256, BLACK=[255, 255, 255])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img


def getParsingLoss(model, img_1, img_2):
    criterionParsingLoss = ParsingCrossEntropyLoss(tensor=Tensor)
    L1Loss = torch.nn.L1Loss()
    model = model.cuda(2)
    model.eval()
    img_1 = trans2Tensor(img_1).cuda(2)
    img_2 = trans2Tensor(img_2).cuda(2)

    img_1_parsing = model.forward(Variable(img_1, volatile=False))
    img_2_parsing = model.forward(Variable(img_2, volatile=False))
    img_1_parsing = F.softmax(img_1_parsing, dim=1)
    img_2_parsing = F.softmax(img_2_parsing, dim=1)
    # img_1_p = parsing2Image(img_1_parsing)
    # img_1_p = Image.fromarray(np.uint8(img_1_p))
    # img_1_p.show()


    loss_G_parsing = criterionParsingLoss(img_1_parsing, img_2_parsing).data.cpu().float().numpy()
    loss_G_parsing_L1 = L1Loss(Variable(img_1_parsing.data), Variable(img_2_parsing.data))
    print loss_G_parsing_L1.data.cpu().float().numpy()
    return loss_G_parsing


model = ResGenerator(3, 20)
load_network(model, save_dir, 'G', which_epoch)


img_1_path = "../../datasets/deepfashion/paper_images/256/JOINT/deepfasion/00013/id_00000390/09_1_front.jpg"
#img_2_path = "../../datasets/deepfashion/paper_images/256/JOINT/deepfasion/00013/id_00000390/09_1_front.jpg"
img_2_path = "../../datasets/deepfashion/paper_images/256/JOINT/deepfasion/00013/id_00000390/09_2_side.jpg"
#img_2_path = "../datasets/deepfashion/paper_images/256/JOINT/deepfasion/00013/00013_cvpr.png"

img_1 = Image.open(img_1_path).convert('RGB')
img_2 = Image.open(img_2_path).convert('RGB')
result = getParsingLoss(model, img_1, img_2)
print(result)
