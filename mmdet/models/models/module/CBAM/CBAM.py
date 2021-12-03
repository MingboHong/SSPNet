import torch
import torch.nn as nn
import numpy as np
from ..Highlight_Attention.Attention import SSFAttBlock
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
import mmcv
import cv2
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    # self.gama = 1e-5

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CBAM_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_planes)
        self.ca = ChannelAttention(out_planes)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # residual = x
        x = self.conv_1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.ca(x) * x
        # x = self.sa(x) * x
        sa_out = self.sa(x)

        # att = (sa_out.cpu().numpy()[0])*255
        # att = att.astype(np.uint8).transpose(1, 2, 0)
        # mmcv.imshow(att)
        # cv2.waitKey(0)

        return sa_out


class SCAM(nn.Module):
    def __init__(self):
        super(SCAM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.SSFAttBlock = SSFAttBlock()

    def forward(self, x):
        residual = x

        ca_out = self.avg_pool(x)
        ca_out = self.sigmoid(ca_out)
        x = ca_out * x

        sa_out = torch.mean(x, dim=1, keepdim=True)
        sa_out = self.sigmoid(sa_out)
        # sa_out = 1 - sa_out

        x = sa_out * x

        out = residual + x
        return out


class SCAMV2(nn.Module):
    def __init__(self):
        super(SCAMV2, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.SSFAttBlock = SSFAttBlock()
        self.sigma = 1e-4

    def forward(self, x):
        residual = x

        ca_avg_out = self.avg_pool(x)
        ca_max_out = self.max_pool(x)
        ca_avg_out = ca_avg_out - ca_avg_out.mean() / (ca_avg_out.std() + self.sigma)
        ca_max_out = ca_max_out - ca_max_out.mean() / (ca_max_out.std() + self.sigma)
        ca_out = self.sigmoid(ca_avg_out + ca_max_out)
        x = ca_out * x

        sa_avg_out = torch.mean(x, dim=1, keepdim=True)
        sa_max_out, _ = torch.max(x, dim=1, keepdim=True)

        sa_max_out = sa_max_out - sa_max_out.mean() / (sa_max_out.std() + self.sigma)
        sa_avg_out = sa_avg_out - sa_avg_out.mean() / (sa_avg_out.std() + self.sigma)

        sa_out = self.sigmoid(sa_avg_out + sa_max_out)
        # sa_out = 1 - sa_out

        x = sa_out * x

        out = residual + x
        return out


class SuppCBAM_block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SuppCBAM_block, self).__init__()
        self.Conv_1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(in_planes)  # nn.BatchNorm2d(in_planes)
        # self.pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        # self.Relu = nn.ReLU(inplace=True)
        # self.p = 1 - p
        # self.topk = topk
        # self.SSPAtt_block = SSPAttBlock(topk=self.topk)
        # self.SCAtt_block = SCAttBlock(topk=self.topk)
        self.ca = ChannelAttention(out_planes)
        self.sa = SpatialAttention()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        # if self.dcn is not None:
        #     for m in self.modules():
        #         if hasattr(m, 'conv_offset'):
        #             constant_init(m.conv_offset, 0)

    def forward(self, x):
        # residual = x

        x = self.Conv_1(x)
        x = self.BN(x)
        # x = self.Relu(x)
        c_a = self.ca(x)

        # if self.training and np.around(np.random.uniform(0, 1), 1) > self.p:
        #     c_a = self.SCAtt_block(c_a)
        # elif not self.training:
        #     c_a = self.SCAtt_block(c_a)
        x = c_a * x
        s_a = self.sa(x)

        # if self.training and np.around(np.random.uniform(0, 1), 1) > self.p:
        #     s_a = self.SSPAtt_block(s_a)
        # elif not self.training:
        #     s_a = self.SSPAtt_block(s_a)
        # x = s_a * x

        #
        # out = residual + x
        # out = self.Relu(out)
        #

        return s_a


from ..Fusion import ASPP


class Spatial_Attention(nn.Module):
    def __init__(self, inplanes, reduction_ratio=1, fpn_lvl=4):
        super(Spatial_Attention, self).__init__()
        self.fpn_lvl = fpn_lvl
        self.dila_conv = nn.Sequential(nn.Conv2d(inplanes * fpn_lvl // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1),
                                       ASPP(inplanes // reduction_ratio, inplanes // (4 * reduction_ratio)),
                                       nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(inplanes // reduction_ratio),
                                       nn.ReLU(inplace=False)
                                       )
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.upsample_cfg = dict(mode='nearest')
        self.down_conv = nn.ModuleList()
        self.att_conv = nn.ModuleList()
        for i in range(self.fpn_lvl):
            self.att_conv.append(nn.Conv2d(inplanes // reduction_ratio,
                                           1,
                                           kernel_size=3,
                                           stride=1,  # 2 ** i
                                           padding=1))
            if i == 0:
                down_stride = 1
            else:
                down_stride = 2
            self.down_conv.append(
                nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio, kernel_size=3, stride=down_stride,
                          padding=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):

        prev_shape = x[0].shape[2:]
        multi_feats = [x[0]]

        for i in range(1, len(x)):
            pyr_feats_2x = F.interpolate(x[i], size=prev_shape, **self.upsample_cfg)
            multi_feats.append(pyr_feats_2x)

        multi_feats = torch.cat(multi_feats, 1)
        lvl_fea = self.dila_conv(multi_feats)

        multi_atts = []

        for i in range(self.fpn_lvl):
            lvl_fea = self.down_conv[i](lvl_fea)
            lvl_att = self.att_conv[i](lvl_fea)
            multi_atts.append(self.sigmoid(lvl_att))

        #

        # for i in range(self.fpn_lvl):  # self.fpn_lvl
        #     att = (multi_atts[i].detach().cpu().numpy()[0])
        #     # att /= np.max(att)
        #     #att = np.power(att, 0.8)
        #     att = att * 255
        #     att = att.astype(np.uint8).transpose(1, 2, 0)
        #    # att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        #     mmcv.imshow(att)
        #     cv2.waitKey(0)

        return multi_atts


"""

        for i in range(self.fpn_lvl):
            multi_atts.append(self.sigmoid(self.att_conv[i](x)))
        # att = (s_a.cpu().numpy()[0])*255
        # att = att.astype(np.uint8).transpose(1, 2, 0)
        # mmcv.imshow(att)
        # cv2.waitKey(0)

        # att = (s_a.cpu().numpy()[0])*255*5
        # #np.save('../tools/visual/att/att.npy', att)
        # att = att.astype(np.uint8).transpose(1, 2, 0)
        # att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
        # mmcv.imshow(att)
        # cv2.waitKey(0)
        
        
            def cal_att_entroy(self, x):
        b, c, h, w = x.size()
        assert c == 1
        x = x.view(b, c * h * w)
        hist = []
        for i in range(b):
            tem_hist = torch.histc(x[i], bins=2)
            tem_hist = tem_hist[tem_hist > 0]
            propab = tem_hist/tem_hist.sum()
            hist.append((propab*torch.log2(1/propab)).sum().unsqueeze(0))
        hist = torch.cat(hist)
        return hist

"""
