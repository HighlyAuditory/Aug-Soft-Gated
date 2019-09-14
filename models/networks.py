#coding=utf-8

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from .geo.geotnf.transformation import GeometricTnf
from .u_res_net import UResNet, UResNetLast
from .warp_res_net_aff_tps import WarpResGenerator as WarpResGenerator_AFFTPS
from .res_net import ResGenerator
from .u_net import UnetGenerator

def define_G(input_nc, output_nc, which_G='wapResNet_v3_afftps'):
    if which_G == 'wapResNet_v3_afftps':
        netG = WarpResGenerator_AFFTPS(23, input_nc - 23, output_nc)
    elif which_G == 'resNet':
        netG = ResGenerator(input_nc, output_nc)
    elif which_G == 'UNet':
        netG = UnetGenerator(input_nc, output_nc)

    if torch.cuda.is_available():
        netG.cuda()

    netG.apply(weights_init)

    return netG


def define_D(input_nc, getIntermFeat=False, num_D=1):
    netD = MultiscaleDiscriminator(input_nc, num_D=num_D, getIntermFeat=getIntermFeat)
    if torch.cuda.is_available():
        netD.cuda()
    netD.apply(weights_init)

    return netD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        if torch.cuda.is_available():
            self.use_cuda = True
        self.tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=self.use_cuda)
        self.affTnf = GeometricTnf(geometric_model='affine', use_cuda=self.use_cuda)

        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU( ),

            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def warp(self, x, theta_aff, theta_aff_tps):
        #print x.shape
        resizeTgt = GeometricTnf(out_h=x.shape[2], out_w=x.shape[3], use_cuda=self.use_cuda)
        print (theta_aff.shape)
        warped_x_aff = self.affTnf(x, theta_aff.view(-1, 2, 3))
        warped_x_aff_tps = self.tpsTnf(warped_x_aff, theta_aff_tps)

        return resizeTgt(warped_x_aff_tps)

    def forward(self, data):
        b, c, h, w = data.shape
        x = data[:, :c-2, :, :]

        theta_aff = data[:, c-2:c-1, :1, :6].view(b, 6)
        theta_aff_tps = data[:, c-1:, :1, :18].view(b, 18)

        z = x + self.warp(self.main(x), theta_aff, theta_aff_tps)
        z = torch.cat([z, data[:, c-2:, :, :]], dim=1)

        return z




class Generator_warpResNet(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, input_nc_1, input_nc_2, output_nc, conv_dim=64, repeat_num=6):
        super(Generator_warpResNet, self).__init__()

        layers_d = []
        layers_d.append(nn.ReflectionPad2d(3))
        layers_d.append(nn.Conv2d(input_nc_1, conv_dim, kernel_size=7, padding=0))
        layers_d.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_d.append(nn.ReLU( ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(3):
            layers_d.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1))
            layers_d.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers_d.append(nn.ReLU( ))
            curr_dim = curr_dim * 2

        layers_d_2 = []
        layers_d_2.append(nn.ReflectionPad2d(3))
        layers_d_2.append(nn.Conv2d(input_nc_2, conv_dim, kernel_size=7, padding=0))
        layers_d_2.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_d_2.append(nn.ReLU( ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(3):
            layers_d_2.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1))
            layers_d_2.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers_d_2.append(nn.ReLU( ))
            curr_dim = curr_dim * 2

        layers_b = []
        # Bottleneck
        for i in range(repeat_num):
            layers_b.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        layers_u = []
        # Up-Sampling
        curr_dim = curr_dim * 2
        #for i in range(2):
        for i in range(3):
            layers_u.append(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers_u.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers_u.append(nn.ReLU( ))
            curr_dim = curr_dim // 2

        layers_u.append(nn.ReflectionPad2d(3))
        layers_u.append(nn.Conv2d(curr_dim, output_nc, kernel_size=7, padding=0))
        #layers_u.append(nn.ReLU( ))  # replace layers.append(nn.Tanh())
        layers_u.append(nn.Tanh())

        self.down_1 = nn.Sequential(*layers_d)
        self.down_2 = nn.Sequential(*layers_d_2)
        self.blocks = nn.Sequential(*layers_b)
        self.up = nn.Sequential(*layers_u)

    def forward(self, x, theta_aff, theta_aff_tps):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        d1 = self.down_1(x1)
        tmp = torch.cat([d1, theta_aff, theta_aff_tps], dim=1)

        b = self.blocks(tmp)
        _, c, _, _ = b.shape
        b = b[:, :c - 2, :, :]

        d2 = self.down_2(x2)
        d_all = torch.cat([b, d2], dim=1)

        u = self.up(d_all)

        return u

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            # sequence += [nn.Sigmoid()]
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)






class ResidualBlock_resNet(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock_resNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU( ),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Generator_resNet(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, input_nc, output_nc, conv_dim=64, repeat_num=6):
        super(Generator_resNet, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU( ))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU( ))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock_resNet(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU( ))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, output_nc, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
        self.main = self.main.cuda(0)

    def forward(self, x, theta_aff, theta_aff_tps):
        return self.main(x)



class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model = model+[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
