import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d

from U2net import RSU4F, RSU4, RSU5, _upsample_like, RSU6, RSU7
from deconv import DeformConv2D
from typing import Dict, List
from convcrf import GaussCRF, get_test_conf, get_default_conf
from vgg16_bn import ChannelAttention, SpatialAttention




class Self_Correlation_Per(nn.Module):
    # input:[?,512,32,32] out:[?,115,32,32]
    def __init__(self, nb_pools=115):
        super(Self_Correlation_Per, self).__init__()
        self.nb_pools = nb_pools

    def forward(self, x):
        b, c, row, col = x.shape
        # print(b, c, row, col)
        nb_maps = row * col
        xn = F.normalize(x, p=2, dim=1)
        x_corr_3d = torch.matmul(xn.permute(0, 2, 3, 1).view(b, -1, c), xn.view(b, c, -1)) / c
        # print(x_corr_3d.shape)
        if self.nb_pools is not None:
            ranks = torch.round(torch.linspace(1.0, nb_maps - 1, self.nb_pools)).long()
        else:
            ranks = torch.arange(1, nb_maps)
        x_corr = x_corr_3d.view([-1, nb_maps, row, col])
        x_sort, _ = torch.topk(x_corr, k=self.nb_pools,dim=1)
        # x_f1st_sort = x_sort.permute(1, 2, 3, 0)
        # x_f1st_pool = x_f1st_sort[ranks]
        # x_pool = x_f1st_pool.permute(3, 0, 1, 2)

        return x_sort

def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class ZeroWindow:
    def __init__(self):
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1):
        sigma = h * rat_s, w * rat_s
        # c = h * w
        b, c, h2, w2 = x_in.shape
        key = str(x_in.shape) + str(rat_s)
        if key not in self.store:
            ind_r = torch.arange(h2).float()
            ind_c = torch.arange(w2).float()
            ind_r = ind_r.view(1, 1, -1, 1).expand_as(x_in)
            ind_c = ind_c.view(1, 1, 1, -1).expand_as(x_in)

            # center
            c_indices = torch.from_numpy(np.indices((h, w))).float()
            c_ind_r = c_indices[0].reshape(-1)
            c_ind_c = c_indices[1].reshape(-1)

            cent_r = c_ind_r.reshape(1, c, 1, 1).expand_as(x_in)
            cent_c = c_ind_c.reshape(1, c, 1, 1).expand_as(x_in)

            def fn_gauss(x, u, s):
                return torch.exp(-(x - u) ** 2 / (2 * s ** 2))

            gaus_r = fn_gauss(ind_r, cent_r, sigma[0])
            gaus_c = fn_gauss(ind_c, cent_c, sigma[1])
            out_g = 1 - gaus_r * gaus_c
            out_g = out_g.to(x_in.device)
            self.store[key] = out_g
        else:
            out_g = self.store[key]
        out = out_g * x_in
        return out


class Corr(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        # self.h = hw[0]
        # self.w = hw[1]
        self.topk = topk
        self.zero_window = ZeroWindow()

        self.alpha = nn.Parameter(torch.tensor(
            5., dtype=torch.float32))

    def forward(self, x):
        b, c, h1, w1 = x.shape
        h2 = h1
        w2 = w1

        xn = F.normalize(x, p=2, dim=-3)
        x_aff_o = torch.matmul(xn.permute(0, 2, 3, 1).view(b, -1, c),
                               xn.view(b, c, -1))  # h1 * w1, h2 * w2

        # zero out same area corr
        # x_aff = _zero_window(x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(b, h1*w1, h2*w2)
        x_aff = self.zero_window(x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(b, h1*w1, h2*w2)

        x_c = F.softmax(x_aff * self.alpha, dim=-1) * \
            F.softmax(x_aff * self.alpha, dim=-2)
        x_c = x_c.reshape(b, h1, w1, h2, w2)

        xc_o = x_c.view(b, h1*w1, h2, w2)
        val = get_topk(xc_o, k=self.topk, dim=-3)


        return val


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        res1 = torchvision.models.resnet18(pretrained=True).conv1
        res2 = torchvision.models.resnet18(pretrained=True).relu
        res3 = torchvision.models.resnet18(pretrained=True).maxpool
        cnn_temp1 = torchvision.models.resnet18(pretrained=True).layer1

        cnn_temp2 = torchvision.models.resnet18(pretrained=True).layer2

        cnn_temp3 = torchvision.models.resnet18(pretrained=True).layer3

        cnn_temp4 = torchvision.models.resnet18(pretrained=True).layer4

        cnn_temp = torchvision.models.resnet18(pretrained=True)



        self.layer1 = nn.Sequential(res1,res2,res3,cnn_temp1)

        self.layer2 = nn.Sequential(cnn_temp2)

        self.layer3 = nn.Sequential(cnn_temp3)

        self.layer4 = nn.Sequential(cnn_temp4)

        self.ch_att512 = ChannelAttention(512)
        self.ch_att256 = ChannelAttention(256)
        self.ch_att128 = ChannelAttention(128)
        self.ch_att64 = ChannelAttention(64)
        self.corr1 = Corr(topk=32)
        self.corr32 = Corr(topk=32)
        # 32
        self.corr64 = Corr(topk=512)
        self.corr128 = Corr(topk=128)



        self.sam1 = SpatialAttention()
        self.sam2 = SpatialAttention()
        self.sam3 = SpatialAttention()
        self.sam4 = SpatialAttention()
        self.sam5 = SpatialAttention()
        self.sam6 = SpatialAttention()
        self.sam7 = SpatialAttention()




        self.stage5d = RSU4(64, 32, 64)

        self.stage4d = RSU4(192, 96, 128)
        self.stage3d = RSU5(192, 128, 192)
        self.stage2d = RSU5(192, 64, 64)
        self.stage1d = RSU6(64, 32, 32)
        self.stage0d = RSU6(32, 16, 16)
        self.stage7d = RSU6(16, 16, 16)


        # 之后在这里做优化
        self.side1 = nn.Conv2d(192, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(64, 1, 3, padding=1)
        self.side5 = nn.Conv2d(64, 1, 3, padding=1)
        self.side6 = nn.Conv2d(16, 1, 3, padding=1)
        self.side7 = nn.Conv2d(32, 1, 3, padding=1)
        self.side8 = nn.Conv2d(16, 1, 3, padding=1)

        self.outconv = nn.Conv2d(6, 1, 1)

        # self.aspp1 = atrous_spatial_pyramid_pooling(in_channel=230, depth=230, rate_dict=[2, 4, 8, 12])
        self.aspp1 = models.segmentation.deeplabv3.ASPP(in_channels=32, out_channels=32,atrous_rates=[4, 8, 12, 16])
        # 4 8 12 16 == 0.53

        self.bn1 = nn.BatchNorm2d(230)
        self.conv1 = nn.Sequential(
            nn.Conv2d(230, 230, 1),
            nn.BatchNorm2d(230),
            nn.ReLU(),
        )
        #
        # self.aspp3 = atrous_spatial_pyramid_pooling(in_channel=32, depth=32, rate_dict=[2, 4, 8, 12])
        self.aspp2 = models.segmentation.deeplabv3.ASPP(in_channels=128,out_channels=128, atrous_rates=[4, 8, 12, 16])

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # self.aspp3 = atrous_spatial_pyramid_pooling(in_channel=32, depth=32, rate_dict=[4, 8, 12, 16])
        self.aspp3 = models.segmentation.deeplabv3.ASPP(in_channels=64,out_channels=64, atrous_rates=[4, 8, 12, 16])


        # self.aspp2 = atrous_spatial_pyramid_pooling(in_channel=64, depth=64, rate_dict=[2, 4, 8, 12])
        self.aspp4 = models.segmentation.deeplabv3.ASPP(in_channels=32,out_channels=32, atrous_rates=[4, 8, 12, 16])

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)


    def forward(self, x):
        x0 = x
        x1 = self.layer1(x0)

        x1_cr = self.aspp3(x1)



        x2 = self.layer2(x1)
        # print(x2.shape)


        x2_cr = self.corr128(x2)


        x2_cr = self.aspp2(x2_cr)



        x3 = self.layer3(x2)

        x3_cr = self.corr32(x3)


        x3_cr = self.aspp4(x3_cr)


        # return x3
        x4 = self.layer4(x3)

        x4 = self.corr1(x4)


        x4 = self.aspp1(x4)

        x4 = _upsample_like(x4, x3)

        x4 = self.sam1(torch.cat((x4, x3_cr), 1))


        up4d = self.stage5d(x4)

        up3d_out = _upsample_like(up4d, x2)

        up3d = self.sam2(torch.cat((up3d_out, x2_cr), 1))

        up3d = self.stage4d(up3d)

        up2d_out = _upsample_like(up3d, x1)


        up2d_out = self.sam3(torch.cat((up2d_out, x1_cr), 1))
        up2d = self.stage3d(up2d_out)

        up1d_out = self.upsample2(up2d)

        up1d_out = self.sam4(up1d_out)
        up1d = self.stage2d(up1d_out)

        up1d = self.upsample2(up1d)

        up0d_out = self.sam5(up1d)
        up0d = self.stage1d(up0d_out)

        up0d = self.upsample2(up0d)

        up7d_out = self.sam6(up0d)
        up7d = self.stage0d(up7d_out)


        d5 = self.side6(up7d)
        d5 = _upsample_like(d5, x0)



        d1 = self.side5(up1d)
        d1 = _upsample_like(d1, x0)

        d2 = self.side1(up2d)
        d2 = _upsample_like(d2, x0)



        d3 = self.side3(up3d)
        d3 = _upsample_like(d3, x0)

        d4 = self.side4(up4d)
        d4 = _upsample_like(d4, x0)

        d7 = self.side7(up0d)
        d7 = _upsample_like(d7, x0)

        # d8 = self.side8(up7d)
        # d8 = _upsample_like(d8, x0)

        d0 = self.outconv(torch.cat((d1,d2, d3, d4, d5, d7), 1))

        return F.sigmoid(d0), F.sigmoid(d1),  F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d7)



if __name__ == '__main__':
    x = torch.randn([2, 3, 256, 256])

    net = VGG19()
    # print(net)
    a= net(x)
    print(a[0].shape)
