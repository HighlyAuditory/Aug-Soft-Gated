#coding=utf-8
import torch
import torch.nn as nn
from .geo.geotnf.transformation import GeometricTnf
import torch.nn.functional as F

class UResNet(nn.Module):
    def __init__(self, input_nc_1, input_nc_2, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(UResNet, self).__init__()
        self.n_downsampling = n_downsampling

        activation = nn.ReLU(True)

        ### first downsample 1
        first_down_1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.add_module('first_down_1', nn.Sequential(first_down_1[0], first_down_1[1], first_down_1[2], first_down_1[3]))

        first_down_2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.add_module('first_down_2', nn.Sequential(first_down_2[0], first_down_2[1], first_down_2[2], first_down_2[3]))


        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            down = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
            self.add_module('down%d' % i, nn.Sequential(down[0], down[1], down[2]))

        ### mid resnet blocks
        mid_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            mid_blocks += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.add_module('model_blocks', nn.Sequential(mid_blocks[0], mid_blocks[1], mid_blocks[2], mid_blocks[3], mid_blocks[4], \
                                                      mid_blocks[5], mid_blocks[6], mid_blocks[7], mid_blocks[8]))

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if i == 0:
                in_nc, out_nc = ngf * mult, int(ngf * mult / 2)
            else:
                #in_nc, out_nc = ngf * mult * 2, int(ngf * mult / 2)
                in_nc, out_nc = ngf * mult * 3, int(ngf * mult / 2)

            up = [nn.ConvTranspose2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
                  norm_layer(int(ngf * mult / 2)),
                  activation]
            self.add_module('up%d' % i, nn.Sequential(up[0], up[1], up[2]))

        ### last upsample
        last_up = [nn.ReflectionPad2d(3),
                  #nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Conv2d(ngf * 3, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.add_module('last_up', nn.Sequential(last_up[0], last_up[1], last_up[2]))

        self.tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=True)
        self.affTnf = GeometricTnf(geometric_model='affine', use_cuda=True)

    def warp(self, x, theta_aff, theta_aff_tps):

        ### 这里加多几个块（类型DREAM），而不是简单的变形，而是在深层的更深层进行变形。
        ### 这个得怎么加呢？
        _, _, h, w = x.shape
        resizeTgt = GeometricTnf(out_h=h, out_w=w, use_cuda=True)
        warped_x_aff = self.affTnf(x, theta_aff.view(-1, 2, 3))
        warped_x_aff_tps = self.tpsTnf(warped_x_aff, theta_aff_tps)

        return resizeTgt(warped_x_aff_tps)


    def forward(self, x, theta_aff, theta_aff_tps):
        b, c, h, w = x.shape
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        theta_aff = theta_aff[:, :, :1, :6].view(b, 6)
        theta_aff_tps = theta_aff_tps[:, :, :1, :18].view(b, 18)

        # branch 1
        first_down_feature_1 = getattr(self, 'first_down_1')(x1)
        down_feat_1 = first_down_feature_1
        down_features_1 = [down_feat_1]
        for i in range(self.n_downsampling):
            down_feat_1 = getattr(self, 'down%d' % i)(down_feat_1)
            down_features_1.append(down_feat_1)

        # 对branch1 每层特征做warp
        for i in range(len(down_features_1)):
            down_features_1[i] = self.warp(down_features_1[i], theta_aff, theta_aff_tps)

        # branch 2
        first_down_feature_2 = getattr(self, 'first_down_2')(x2)
        down_feat_2 = first_down_feature_2
        down_features_2 = [down_feat_2]
        for i in range(self.n_downsampling):
            down_feat_2 = getattr(self, 'down%d' % i)(down_feat_2)
            down_features_2.append(down_feat_2)

        mid_blocks_feature_2 = getattr(self, 'model_blocks')(down_feat_2)

        up_feat_2 = mid_blocks_feature_2
        for i in range(self.n_downsampling):
            up_feat_2 = getattr(self, 'up%d' % i)(up_feat_2)        # 1024
            up_feat_2 = torch.cat([up_feat_2, down_features_2[-(i+2)]], 1)
            up_feat_2 = torch.cat([up_feat_2, down_features_1[-(i+2)]], 1)

        last_upfeature = up_feat_2
        last_upfeature = getattr(self, 'last_up')(last_upfeature)

        return last_upfeature

class UResNetLast(nn.Module):
    def __init__(self, input_nc_1, input_nc_2, output_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=nn.InstanceNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(UResNetLast, self).__init__()
        self.n_downsampling = n_downsampling

        activation = nn.ReLU(True)

        ### first downsample 1
        first_down_1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0),
                        norm_layer(ngf), activation]
        self.add_module('first_down_1',
                        nn.Sequential(first_down_1[0], first_down_1[1], first_down_1[2], first_down_1[3]))

        first_down_2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0),
                        norm_layer(ngf), activation]
        self.add_module('first_down_2',
                        nn.Sequential(first_down_2[0], first_down_2[1], first_down_2[2], first_down_2[3]))

        self.warpblock = nn.Sequential(*[ResidualBlock(dim_in=ngf, dim_out=ngf)])

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            down = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult * 2), activation]
            self.add_module('down%d' % i, nn.Sequential(down[0], down[1], down[2]))

        ### mid resnet blocks
        mid_blocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            mid_blocks += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.add_module('model_blocks',
                        nn.Sequential(mid_blocks[0], mid_blocks[1], mid_blocks[2], mid_blocks[3], mid_blocks[4], \
                                      mid_blocks[5], mid_blocks[6], mid_blocks[7], mid_blocks[8]))

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if i == 0:
                in_nc, out_nc = ngf * mult, int(ngf * mult / 2)
            else:
                # in_nc, out_nc = ngf * mult * 2, int(ngf * mult / 2)
                in_nc, out_nc = ngf * mult * 3, int(ngf * mult / 2)

            up = [nn.ConvTranspose2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
                  norm_layer(int(ngf * mult / 2)),
                  activation]
            self.add_module('up%d' % i, nn.Sequential(up[0], up[1], up[2]))

        ### last upsample
        last_up = [nn.ReflectionPad2d(3),
                   # nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Conv2d(ngf * 3, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]
        self.add_module('last_up', nn.Sequential(last_up[0], last_up[1], last_up[2]))


    def forward(self, x, theta_aff, theta_aff_tps):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        # branch 1
        first_down_feature_1 = getattr(self, 'first_down_1')(x2)
        down_feat_1 = first_down_feature_1
        down_features_1 = [down_feat_1]
        for i in range(self.n_downsampling):
            down_feat_1 = getattr(self, 'down%d' % i)(down_feat_1)
            down_features_1.append(down_feat_1)

        # branch 2
        first_down_feature_2 = getattr(self, 'first_down_2')(x1)
        b, c, h, w = first_down_feature_2.shape
        theta_aff = theta_aff.view(b, -1, h, w)
        theta_aff_tps= theta_aff_tps.view(b, -1, h, w)
        _, c2, _, _ = theta_aff.shape
        tmp = torch.cat([first_down_feature_2, theta_aff, theta_aff_tps], dim=1)
        warpblock_feature = self.warpblock(tmp)

        #down_feat_2 = first_down_feature_2
        down_feat_2 = warpblock_feature
        down_features_2 = [down_feat_2]
        for i in range(self.n_downsampling):
            down_feat_2 = getattr(self, 'down%d' % i)(down_feat_2)
            down_features_2.append(down_feat_2)

        # 放在N block之前
        # tmp = torch.cat([down_feat_2, theta_aff, theta_aff_tps], dim=1)
        # warpblock_feature = self.warpblock(tmp)

        mid_blocks_feature_2 = getattr(self, 'model_blocks')(down_feat_2)

        up_feat_2 = mid_blocks_feature_2
        for i in range(self.n_downsampling):
            up_feat_2 = getattr(self, 'up%d' % i)(up_feat_2)  # 1024
            up_feat_2 = torch.cat([up_feat_2, down_features_2[-(i + 2)]], 1)
            up_feat_2 = torch.cat([up_feat_2, down_features_1[-(i + 2)]], 1)

        last_upfeature = up_feat_2
        last_upfeature = getattr(self, 'last_up')(last_upfeature)

        return last_upfeature

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


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        if torch.cuda.is_available():
            self.use_cuda = True
        # self.tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=self.use_cuda)
        # self.affTnf = GeometricTnf(geometric_model='affine', use_cuda=self.use_cuda)

        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    # def warp(self, x, theta_aff, theta_aff_tps):
    #     resizeTgt = GeometricTnf(out_h=x.shape[2], out_w=x.shape[3], use_cuda=self.use_cuda)
    #     warped_x_aff = self.affTnf(x, theta_aff.view(-1, 2, 3))
    #     warped_x_aff_tps = self.tpsTnf(warped_x_aff, theta_aff_tps)
    #
    #     return resizeTgt(warped_x_aff_tps)

    def grid_sample(self, x, theta_aff_grid, theta_aff_tps_grid):
        resizeTgt = GeometricTnf(out_h=x.shape[2], out_w=x.shape[3], use_cuda=self.use_cuda)
        b, c, h, w = theta_aff_grid.shape
        theta_aff_tmp = theta_aff_grid.view(-1, c*h*w)[:, :240*240*2]
        theta_aff_grid = theta_aff_tmp.view(b, 240, 240, 2)
        theta_aff_tps_tmp = theta_aff_tps_grid.view(-1, c*h*w)[:, :240*240*2]
        theta_aff_tps_grid = theta_aff_tps_tmp.view(b, 240, 240, 2)

        warped_x_aff = F.grid_sample(x, theta_aff_grid)
        warped_x_aff_tps = F.grid_sample(warped_x_aff, theta_aff_tps_grid)

        return resizeTgt(warped_x_aff_tps)



    def forward(self, data):
        b, c, h, w = data.shape
        x = data[:, :c-2*2, :, :]

        theta_aff = data[:, c-2*2:c-2, :, :]
        theta_aff_tps = data[:, c-2:, :, :]

        #z = x + self.warp(self.main(x), theta_aff, theta_aff_tps)
        z = x + self.grid_sample(self.main(x), theta_aff, theta_aff_tps)
        #z = torch.cat([z, data[:, c-2:, :, :]], dim=1)

        return z