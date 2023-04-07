import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from matplotlib import pyplot
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d
from torchvision import transforms

from U2net import RSU4F, RSU4, RSU5, _upsample_like, RSU6
from deconv import DeformConv2D
from typing import Dict, List
from convcrf import GaussCRF, get_test_conf, get_default_conf
from vgg16_bn import  Self_Correlation_Per, atrous_spatial_pyramid_pooling, SpatialAttention, \
    Self_Correlation_zero, DoubleConv


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


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
        x_aff = self.zero_window(x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(b, h1 * w1, h2 * w2)

        x_c = F.softmax(x_aff * self.alpha, dim=-1) * \
              F.softmax(x_aff * self.alpha, dim=-2)
        x_c = x_c.reshape(b, h1, w1, h2, w2)

        xc_o = x_c.view(b, h1 * w1, h2, w2)
        val = get_topk(xc_o, k=self.topk, dim=-3)

        return val


def non_local(x, ind):
    b, c, h2, w2 = x.shape
    b, _, h1, w1 = ind.shape

    x = x.reshape(b, c, -1)
    ind = ind.reshape(b, h2 * w2, h1 * w1)
    out = torch.bmm(x, ind).reshape(b, c, h1, w1)
    return out


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        cnn_temp = torchvision.models.vgg16_bn(pretrained=True).features
        # print(cnn_temp)
        # 6  13  26  39
        # 7  14  24  34

        self.layer1 = nn.Sequential(cnn_temp[:7])

        self.layer2 = nn.Sequential(cnn_temp[7:14])

        self.layer3 = nn.Sequential(cnn_temp[14:24])

        self.layer4 = nn.Sequential(cnn_temp[24:33])

        # self.ch_att512 = ChannelAttention(512)
        # self.ch_att256 = ChannelAttention(256)
        # self.ch_att128 = ChannelAttention(128)
        # self.ch_att64 = ChannelAttention(64)
        # self.corr1 = Corr(topk=230)
        # self.corr32 = Corr(topk=32)
        # # 32
        # self.corr64 = Corr(topk=32)
        # self.corr128 = Corr(topk=64)

        self.corr1 = Self_Correlation_Per(nb_pools=230)
        self.corr32 = Self_Correlation_Per(nb_pools=32)
        # 32
        self.corr64 = Self_Correlation_Per(nb_pools=32)
        self.corr128 = Self_Correlation_Per(nb_pools=64)
        # 64

        #
        self.sam1 = SpatialAttention()
        self.sam2 = SpatialAttention()
        self.sam3 = SpatialAttention()
        self.sam4 = SpatialAttention()

        # self.stage5d = RSU4F(371, 64, 128)
        self.stage5d = RSU4F(292, 64, 115)
        # self.stage4d = RSU4(256, 32, 64)
        # self.stage3d = RSU5(128, 16, 32)
        self.stage4d = RSU4(147, 32, 64)
        self.stage3d = RSU5(128, 32, 32)
        self.stage2d = RSU6(32, 8, 16)

        # 之后在这里做优化
        self.side1 = nn.Conv2d(32, 1, 3, padding=1)
        self.side3 = nn.Conv2d(64, 1, 3, padding=1)
        self.side4 = nn.Conv2d(115, 1, 3, padding=1)
        self.side5 = nn.Conv2d(16, 1, 3, padding=1)

        self.outconv = nn.Conv2d(4, 1, 1)

        self.aspp1 = atrous_spatial_pyramid_pooling(in_channel=230, depth=230, rate_dict=[2, 4, 8, 12])
        # self.aspp1 = models.segmentation.deeplabv3.ASPP(in_channels=230, out_channels=230,atrous_rates=[4, 8, 12, 16])
        # 4 8 12 16 == 0.53
        #
        self.bn1 = nn.BatchNorm2d(230)
        self.conv1 = nn.Sequential(
            nn.Conv2d(230, 230, 1),
            nn.BatchNorm2d(230),
            nn.ReLU(),
        )

        self.aspp3 = atrous_spatial_pyramid_pooling(in_channel=32, depth=32, rate_dict=[2, 4, 8, 12])
        # self.aspp3 = models.segmentation.deeplabv3.ASPP(in_channels=32,out_channels=32, atrous_rates=[2, 4, 8, 12])

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # self.aspp3 = atrous_spatial_pyramid_pooling(in_channel=32, depth=32, rate_dict=[4, 8, 12, 16])
        # # self.aspp3 = models.segmentation.deeplabv3.ASPP(in_channels=32,out_channels=32, atrous_rates=[2, 4, 8, 12])
        #
        # self.bn3 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32, 32, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )

        self.aspp2 = atrous_spatial_pyramid_pooling(in_channel=64, depth=64, rate_dict=[2, 4, 8, 12])
        # self.aspp2 = models.segmentation.deeplabv3.ASPP(in_channels=64,out_channels=64, atrous_rates=[2, 4, 8, 12])

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.aspp4 = atrous_spatial_pyramid_pooling(in_channel=64, depth=64, rate_dict=[2, 4, 8, 12])

        self.eca3 = eca_layer(channel=32)
        self.eca2 = eca_layer(channel=64)
        self.eca1 = eca_layer(channel=230)

        # self.val_conv1 = nn.Sequential(
        #     nn.Conv2d(230, 115, 3, padding=1),
        #     nn.BatchNorm2d(115),
        #     nn.ReLU(),
        #     nn.Conv2d(115, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, padding=1),
        #     nn.Conv2d(32, 1, 1),
        #     nn.Sigmoid()
        # )
        # self.val_conv2 = nn.Sequential(
        #     nn.Conv2d(64, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 16, 3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.Conv2d(16, 1, 1),
        #     nn.Sigmoid()
        # )
        # self.val_conv3 = nn.Sequential(
        #     nn.Conv2d(32, 16, 3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.Conv2d(16, 1, 1),
        #     nn.Sigmoid()
        # )
        # # self.no_local = non_local()
        # self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x0 = x
        x1 = self.layer1(x0)
        x1_a = self.aspp4(x1)

        # x1_cr,_ = self.corr32(x1)
        # x1_cr = self.bn3(x1_cr)
        # x1_cr = self.conv3(x1_cr)
        #
        # x1_cr = self.aspp3(x1_cr)

        x2 = self.layer2(x1)

        # x2 = self.ch_att128(x2)
        # x2_cr, ind = self.corr64(x2)

        x2_cr = self.corr64(x2)
        x2_cr = self.bn3(x2_cr)
        # x2_cr = self.eca3(x2_cr)
        x2_cr = self.conv3(x2_cr)
        # x2_cr = self.eca3(x2_cr)
        x2_cr = self.aspp3(x2_cr)

        x3 = self.layer3(x2)

        # x3 = self.ch_att256(x3)
        # x3_cr, ind= self.corr128(x3)

        x3_cr = self.corr128(x3)
        x3_cr = self.bn2(x3_cr)
        # x3_cr = self.eca2(x3_cr)
        x3_cr = self.conv2(x3_cr)
        # x3_cr = self.eca2(x3_cr)
        x3_cr = self.aspp2(x3_cr)

        # return x3
        x4 = self.layer4(x3)
        # print(x4.shape)
        # x4 = self.ch_att512(x4)
        # x4, ind = self.corr1(x4)

        x4 = self.corr1(x4)
        x4 = self.bn1(x4)
        # x4 = self.eca1(x4)
        x4 = self.conv1(x4)

        x4 = self.aspp1(x4)

        # x4_u = _upsample_like(x4, x3)
        #
        x4 = self.sam1(torch.cat((x4, x3_cr), 1))

        up4d = self.stage5d(x4)
        # up4d = self.stage5d(torch.cat((x4, x3_cr), 1))
        up3d_out = _upsample_like(up4d, x2)
        #
        up3d = self.sam2(torch.cat((up3d_out, x2_cr), 1))
        # up3d = self.stage4d(torch.cat((up3d_out, x2_cr), 1))

        up3d = self.stage4d(up3d)
        # up3d = self.stage4d(torch.cat((up3d_out, x2_cr), 1))
        up2d_out = _upsample_like(up3d, x1)
        #
        # # up2d = self.stage3d(torch.cat((up2d_out, x1_cr), 1))
        up2d_out = self.sam3(torch.cat((up2d_out, x1_a), 1))

        up2d = self.stage3d(up2d_out)
        # up2d = self.stage3d(torch.cat((up2d_out, x1_a), 1))

        up1d_out = _upsample_like(up2d, x0)
        #
        up1d_out = self.sam4(up1d_out)
        up1d = self.stage2d(up1d_out)

        d1 = self.side5(up1d)

        #
        # up1d_out = self.sam4(up1d_out)
        # up1d = self.stage2d(up1d_out)

        d2 = self.side1(up2d)
        d2 = _upsample_like(d2, x0)

        d3 = self.side3(up3d)
        d3 = _upsample_like(d3, x0)

        d4 = self.side4(up4d)
        d4 = _upsample_like(d4, x0)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = torchvision.transforms.Compose([transforms.ToPILImage(), transforms.Resize((320, 320), interpolation=2),
                                                torchvision.transforms.ToTensor(),
                                                # transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),
                                                ])

    image = cv2.imread('D:/code/CSFD/CoMoFoD_small_v2/173_F.png')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).cuda()

    model = VGG19()

    model.load_state_dict(torch.load('log_vunt_vgg/model_18_94000_0.31395423412323.pth'))
    model = model.to(device)
    model.eval()

    # print(net)
    pred, _, _, _, _ = model(img)
    pred = pred.squeeze(0)
    # print(pred[1])

    pred = pred.squeeze(0)

    # pred  = torch.sigmoid(pred)
    #
    # pred = pred.cpu().detach().numpy()
    # pred = pred > 0.5
    # # pyplot.imshow(pred)
    # # pyplot.show()
    # pyplot.imsave("C:/Users/Administrator/PycharmProjects/busternet_pytorch/result/173xiaome.png", pred, )

