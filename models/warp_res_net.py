#coding=utf-8
import torch
import torch.nn as nn
from .geo.geotnf.transformation import GeometricTnf
import torch.nn.functional as F


class WarpResGenerator(nn.Module):
    def __init__(self, input_nc_1, input_nc_2, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(WarpResGenerator, self).__init__()
        activation = nn.ReLU(True)

        ### downsample 1
        down_1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            down_1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### downsample 2
        down_2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            down_2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mid_block = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            mid_block += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        up = []
        for i in range(n_downsampling):
            mult = (2 ** (n_downsampling - i)) * 2
            up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * mult / 2)), activation]

        up += [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            norm_layer(ngf), activation]

        up += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.down_1 = nn.Sequential(*down_1)
        self.down_2 = nn.Sequential(*down_2)
        self.warpblock = nn.Sequential(*[WarpResBlock(dim_in=ngf*8, dim_out=ngf*8)])
        self.mid_block = nn.Sequential(*mid_block)
        self.up = nn.Sequential(*up)

    def forward(self, x, theta_aff, theta_tps, theta_aff_tps, policy_binary):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        x1_feature = self.down_1(x1)
        x2_feature = self.down_2(x2)

        b, c, h, w = x1_feature.shape
        theta_aff = theta_aff.view(b, -1, h, w)
        theta_tps = theta_tps.view(b, -1, h, w)
        theta_aff_tps = theta_aff_tps.view(b, -1, h, w)
        policy_binary = policy_binary.view(b, -1, h, w)
        tmp = torch.cat([x1_feature, theta_aff, theta_tps, theta_aff_tps, policy_binary], dim=1)

        warpblock_feature = self.warpblock(tmp)

        cat_feature = torch.cat([x2_feature, warpblock_feature], dim=1)
        mid_feature = self.mid_block(cat_feature)
        up_feature = self.up(mid_feature)

        return up_feature


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


class WarpResBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(WarpResBlock, self).__init__()
        if torch.cuda.is_available():
            self.use_cuda = True

    def grid_sample_by_policy(self, x, theta_aff_grid, theta_tps_grid, theta_aff_tps_grid, policy_binary):
        resizeTgt = GeometricTnf(out_h=x.shape[2], out_w=x.shape[3], use_cuda=self.use_cuda)

        b, c, h, w = theta_aff_grid.shape
        theta_aff_tmp = theta_aff_grid.view(-1, c*h*w)[:, :240*240*2]
        theta_aff_grid = theta_aff_tmp.view(b, 240, 240, 2)

        theta_tps_tmp = theta_tps_grid.view(-1, c*h*w)[:, :240*240*2]
        theta_tps_grid = theta_tps_tmp.view(b, 240, 240, 2)

        theta_aff_tps_tmp = theta_aff_tps_grid.view(-1, c*h*w)[:, :240*240*2]
        theta_aff_tps_grid = theta_aff_tps_tmp.view(b, 240, 240, 2)

        policy = policy_binary.view(-1, policy_binary.shape[1]*h*w)[:, 0:2]

        ## 这里应该做 两个block序列，然后通过action_mask来控制。
        block_1_action = policy[:, 0].contiguous()
        block_2_action = policy[:, 1].contiguous()
        action_1_mask = block_1_action.float().view(-1, 1, 1, 1)
        action_2_mask = block_2_action.float().view(-1, 1, 1, 1)

        ### block 1
        if action_1_mask.data.sum() == 0:
            x = x
        else:
            aff_x = F.grid_sample(x, theta_aff_grid)
            aff_x = resizeTgt(aff_x)
            x = aff_x * action_1_mask + x * (1 - action_1_mask)

        ### block 2
        if action_2_mask.data.sum() == 0:
            x = x
        else:
            tps_x = F.grid_sample(x, theta_tps_grid)
            aff_tps_x = F.grid_sample(x, theta_aff_tps_grid)
            merge_x = tps_x * (1 - action_1_mask) + aff_tps_x * action_1_mask
            merge_x = resizeTgt(merge_x)

            x = merge_x * action_2_mask + x * (1 - action_2_mask)

        return x


    def forward(self, data):
        ngf = 64
        b, c, h, w = data.shape
        x = data[:, :c-64*7, :, :]
        theta_aff = data[:, c-ngf*7:c-ngf*5, :, :]
        theta_tps = data[:, c-ngf*5:c-ngf*3, :, :]
        theta_aff_tps = data[:, c-ngf*3:c-ngf, :, :]
        policy_binary = data[:, c-ngf:, :, :]

        z = x + self.grid_sample_by_policy(x, theta_aff, theta_tps, theta_aff_tps, policy_binary)

        return z