import torch.nn as nn
import torch
from torch.autograd import Variable
import math

def main():
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = mynet()
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    x = torch.autograd.Variable(torch.FloatTensor(64, 512, 15, 15).cuda())  # batch_sizexCxWxH
    out = net(x)
    print(out.size())

class SpatialPool(nn.Module):
    def __init__(self, amd0=225, kd=3):
        super(SpatialPool, self).__init__()
        print('*** spatial_pooling.py : __init__() ***', amd0,kd)
        self.use_gpu = True
        self.amd0 = amd0 #225
        self.kd = kd
        self.padding = nn.ReplicationPad2d(1).cuda()

        ww = hh = int(math.sqrt(amd0)) ## 15
        self.counts = torch.LongTensor(amd0,kd*kd) ## size [225,9]
        v = [[(hh+2)*i + j for j in range(ww+2)] for i in range(hh+2)]
        count = 0
        for h in range(1,hh+1):
            for w in range(1,ww+1):
                self.counts[count,:] = torch.LongTensor([v[h - 1][w - 1], v[h - 1][w], v[h - 1][w + 1],
                                                    v[h][w - 1], v[h][w], v[h][w + 1],
                                                    v[h + 1][w - 1], v[h + 1][w], v[h + 1][w + 1]])
                count += 1

        self.counts =self.counts.cuda(0)
        #self.register_buffer("counts", counts)

    def forward(self, fm):
        fm = self.padding(fm) ## FM is Variable of size[batch_size,512,15,15]
        fm = fm.permute(0, 2, 3, 1).contiguous()
        fm = fm.view(fm.size(0), -1, fm.size(3))
        print('fm size and max ', fm.size(), torch.max(self.counts))
        pfm = fm.index_select(1,Variable(self.counts[:,0]))
        for h in range(1,self.kd*self.kd):
            pfm = torch.cat((pfm,fm.index_select(1, Variable(self.counts[:, h]))),2)
        # print('pfm size:::::::: ', pfm.size()) #[batch_size,225,512*9]
        return pfm


class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.cl = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.featPool = SpatialPool(amd0=225)

    def forward(self, x):
        x = self.cl(x)
        x = self.featPool(x)
        return x

if __name__ == '__main__':
    main()