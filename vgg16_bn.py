import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from typing import Dict, List




class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ZeroWindow:
    def __init__(self):
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1):
        sigma = h * rat_s, w * rat_s
        # c = h * w
        b, c, h2, w2 = x_in.shape
        key = str(x_in.shape) + str(rat_s)
        if key not in self.store:
            # 高宽
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


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 228) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        y = F.sigmoid(out)
        return x * y


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

class Self_Correlation_zero(nn.Module):
    # input:[?,512,32,32] out:[?,115,32,32]
    def __init__(self, nb_pools=115):
        super(Self_Correlation_zero, self).__init__()
        self.nb_pools = nb_pools
        self.zero_window = ZeroWindow()

    def forward(self, x):
        b, c, row, col = x.shape
        nb_maps = row * col
        xn = F.normalize(x, p=2, dim=1)
        x_corr_3d = torch.matmul(xn.permute(0, 2, 3, 1).view(b, -1, c), xn.view(b, c, -1)) / c
        # print(x_corr_3d.shape, "1")
        x_aff = self.zero_window(x_corr_3d.view(b, -1, row, col), row, col, rat_s=0.05).reshape(b, row*col, row*col)
        # print(x_aff.shape, "2")
        if self.nb_pools is not None:
            ranks = torch.round(torch.linspace(1.0, nb_maps - 1, self.nb_pools)).long()
        else:
            ranks = torch.arange(1, nb_maps)
        x_corr = x_aff.view([-1, nb_maps, row, col])
        x_sort, _ = torch.topk(x_corr, k=nb_maps, sorted=True, dim=1)
        x_f1st_sort = x_sort.permute(1, 2, 3, 0)
        x_f1st_pool = x_f1st_sort[ranks]
        x_pool = x_f1st_pool.permute(3, 0, 1, 2)

        return x_pool
class Self_Correlation_Per_tt(nn.Module):
    # input:[?,512,32,32] out:[?,115,32,32]
    def __init__(self, nb_pools=115):
        super(Self_Correlation_Per_tt, self).__init__()
        self.nb_pools = nb_pools
        self.nb_pools2 = 9

        self.kernel_size = 8
        self.stride = 1


    def forward(self, x):
        # print(x.shape)
        b, c, row, col = x.shape
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches = patches.contiguous().view(b, c, -1, self.kernel_size, self.kernel_size)
        b2, c2, c2_2, row2, col2 = patches.shape
        nb_maps_2 = row2 * col2
        patches = F.normalize(patches, p=2, dim=2)
        x_corr_3d_2 = torch.matmul(patches.permute(0, 1, 3, 4, 2).view(b2, c2, -1, c2_2), patches.view(b2, c2, c2_2, -1)) / c2_2
        ranks_2 = torch.round(torch.linspace(1.0, nb_maps_2 - 1, self.nb_pools2)).long()
        x_corr_2 = x_corr_3d_2.view([b2,c2, nb_maps_2, row2, col2])
        x_sort_2, _ = torch.topk(x_corr_2, k=nb_maps_2, sorted=True, dim=2)
        x_f1st_sort_2 = x_sort_2.permute( 2, 3, 4, 0, 1)
        x_f1st_pool_2 = x_f1st_sort_2[ranks_2]
        x_pool_2 = x_f1st_pool_2.permute(3,4, 0, 1, 2)
        x_pool_2 = x_pool_2.contiguous().view(b, c, 24, 24)
        # print(x_pool_2.shape)

        b, c, row, col = x_pool_2.shape
        nb_maps = row * col
        xn = F.normalize(x_pool_2, p=2, dim=1)
        x_corr_3d = torch.matmul(xn.permute(0, 2, 3, 1).view(b, -1, c), xn.view(b, c, -1)) / c
        ranks = torch.round(torch.linspace(1.0, nb_maps - 1, self.nb_pools)).long()

        x_corr = x_corr_3d.contiguous().view([-1, nb_maps, row, col])
        x_sort, _ = torch.topk(x_corr, k=nb_maps, sorted=True, dim=1)
        x_f1st_sort = x_sort.permute(1, 2, 3, 0)
        x_f1st_pool = x_f1st_sort[ranks]
        x_pool = x_f1st_pool.permute(3, 0, 1, 2)
        # print(x_pool.shape)

        return x_pool


class SpatialAttention(nn.Module):
    def __init__(self, ):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = F.sigmoid(x)
        return input + input * y


class atrous_spatial_pyramid_pooling(nn.Module):
    def __init__(self, in_channel, depth=128, rate_dict=[2, 4, 8, 12]):
        super(atrous_spatial_pyramid_pooling, self).__init__()

        self.modules = []
        for index, n_rate in enumerate(rate_dict):
            self.modules.append(nn.Sequential(
                nn.Conv2d(in_channel, depth, 3, dilation=n_rate, padding=n_rate, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(),
                nn.Conv2d(depth, int(depth / 4), 1, dilation=n_rate, bias=False),
                nn.BatchNorm2d(int(depth / 4)),
                nn.ReLU()
            ))

        self.convs = nn.ModuleList(self.modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        return torch.cat(res, 1)

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DctConv(nn.Sequential):
    def __init__(self, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DctConv, self).__init__(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )





cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],}


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



