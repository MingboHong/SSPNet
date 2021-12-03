import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2


def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=1, kernel_size=5, sigma=-1):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = gauss(kernel_size, sigma)
        # kernel = [[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
        #           [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
        #           [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
        #           [0.01330621, 0.0596343, 0.09832033, 0.0596343, 0.01330621],
        #           [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = torch.repeat_interleave(kernel, self.channels, dim=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=3, groups=self.channels)
        return x


class SSPAttBlock(nn.Module):
    def __init__(self, topk=0.2, p=0.5):
        super(SSPAttBlock, self).__init__()
        self.bin_size = 2
        self.thresold = 0.45
        assert 0 < topk < 1, ValueError("Out of Range(0,1)")
        self.topk = topk
        self.thr_area = 4
        self.num_bins = 50
        self.gate = torch.distributions.Bernoulli(p)
        self.exp = 0.5

        self.guassian_conv = GaussianBlurConv(1)

    # def get_topk_idx(self, Att_map):
    #     self.avg_pooling = nn.AvgPool2d((self.bin_size, self.bin_size), stride=self.bin_size)
    #     avg_map = self.avg_pooling(Att_map)
    #     tmp_map = avg_map.view(-1)
    #
    #     K = np.clip(int(len(tmp_map) * self.topk), a_min=1,
    #                 a_max=Att_map.size()[0] * Att_map.size()[1])
    #     _, index = torch.topk(tmp_map, K)
    #     return index, avg_map.size()
    #
    # def SSpAttMap(self, index, block_shape, Att_map):
    #     coords = [[i // block_shape[2], i % block_shape[2]] for i in index]
    #     for coord in coords:
    #         w_s = coord[0] * self.bin_size
    #         w_e = torch.clamp(w_s + self.bin_size, min=w_s, max=Att_map.size()[1])
    #         h_s = coord[1] * self.bin_size
    #         h_e = torch.clamp(h_s + self.bin_size, min=h_s, max=Att_map.size()[2])
    #         Att_mask = Att_map[0, w_s:w_e, h_s:h_e] > 0.3
    #         Att_map[0, w_s:w_e, h_s:h_e][Att_mask] = 0.1 * Att_map[0, w_s:w_e, h_s:h_e][Att_mask]
    #         # Att_map[0, w_s:w_e, h_s:h_e] = 0.1*Att_map[0, w_s:w_e, h_s:h_e]
    #     # Att_map = self.guassian_conv(Att_map.unsqueeze(0)).squeeze(0)
    #     return Att_map

    def Get_SuppAttmap(self, att_map):
        att = att_map.cpu().numpy()
        his = (cv2.calcHist([att], [0], None, [self.num_bins], [0, 1])).squeeze()
        ind_max = his.argsort()[-1:][0]
        ind_sec = his[ind_max + 1:].argsort()[-1:][0] + ind_max + 1
        thresold = ind_sec / self.num_bins

        bw_map = (np.where(att > thresold, 1, 0).squeeze(0) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(bw_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(bw_map)

        for i in range(len(contours)):
            if np.any(hierarchy[0, i, 0:2] != [-1, -1]):
                if cv2.contourArea(contours[i]) > self.thr_area and self.gate.sample() == 1:
                    cv2.drawContours(mask, contours, i, 1, -1)
        mask = mask.astype(np.bool)
        att_map[0, mask] = 0.5 * att_map[0, mask]
        return att_map

    def Get_SuppAttmap_v2(self, att_map):
        att = att_map.cpu().numpy()
        his = (cv2.calcHist([att], [0], None, [self.num_bins], [0, 1])).squeeze()
        ind_max = his.argsort()[-1:][0]
        ind_sec = his[ind_max + 1:].argsort()[-1:][0] + ind_max + 1
        thresold = ind_sec / self.num_bins
        bw_map = (np.where(att > thresold, 1, 0).squeeze(0) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(bw_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            if np.any(hierarchy[0, i, 0:2] != [-1, -1]) and np.all(hierarchy[0, i, -2:-1] == [-1, -1]):
                supp_mask = np.zeros_like(bw_map)
                if self.training:
                    if self.gate.sample() == 1:
                        cv2.drawContours(supp_mask, contours, i, 1, -1)
                        value = self.get_att_value(np.sqrt(cv2.contourArea(contours[i])))
                        supp_mask = supp_mask.astype(np.bool)
                        att_map[0, supp_mask] = torch.pow(att_map[0, supp_mask], 1 / value)

                else:
                    cv2.drawContours(supp_mask, contours, i, 1, -1)
                    value = self.get_att_value(cv2.contourArea(contours[i]))
                    supp_mask = supp_mask.astype(np.bool)
                    att_map[0, supp_mask] = torch.pow(att_map[0, supp_mask], 1 / value)

        return att_map

    def forward(self, att_map):
        assert len(att_map.size()) == 4
        new_map = []
        with torch.no_grad():
            for att in att_map:
                long_side = max(att.size())
                if self.training:
                    scale = 16
                else:
                    scale = 32
                self.bin_size = long_side // scale if long_side // scale > 1 else 1
                self.thr_area = self.bin_size ** 2
                self.min_area = self.thr_area / 100 if self.thr_area / 100 > 1 else 1
                if self.bin_size < 5:
                    new_map.append(att.view(1, att.size()[0],
                                            att.size()[1],
                                            att.size()[2]))
                    continue

                # index, block_shape = self.get_topk_idx(Map)
                # Sp_Att = self.SSpAttMap(index, block_shape, Map)

                supp_map = self.Get_SuppAttmap_v2(att)

                new_map.append(supp_map.view(1, supp_map.size()[0],
                                             supp_map.size()[1],
                                             supp_map.size()[2]))
            new_map = torch.cat(new_map, dim=0)

        return new_map


class SCAttBlock(nn.Module):
    def __init__(self, topk=0.2):
        super(SCAttBlock, self).__init__()

        assert 0 < topk < 1, ValueError("Out of Range(0,1)")
        self.topk = topk

    def get_topk_idx(self, Att_map):
        tmp_map = Att_map.view(-1)
        K = np.clip(int(len(tmp_map) * self.topk), a_min=1,
                    a_max=Att_map.size()[0])
        _, index = torch.topk(tmp_map, K)
        return index

    def ScAttMap(self, index, Att_map):
        Att_map[index] = 1 - Att_map[index]

        return Att_map

    def forward(self, Attention_map):
        assert len(Attention_map.size()) == 4
        new_map = []
        with torch.no_grad():
            for Map in Attention_map:
                index = self.get_topk_idx(Map)
                Sc_Att = self.ScAttMap(index, Map)
                new_map.append(Sc_Att.view(1, Sc_Att.size()[0],
                                           Sc_Att.size()[1],
                                           Sc_Att.size()[2]))
        new_map = torch.cat(new_map, dim=0)
        return new_map


class SSFAttBlock(nn.Module):
    def __init__(self, p=0.5):
        super(SSFAttBlock, self).__init__()
        self.bin_size = 2
        self.thr_area = 4
        self.num_bins = 50
        self.gate = torch.distributions.Bernoulli(p)
        self.exp = 0.5
        self.theta = 30

    def get_att_value(self, area):
        if area < self.thr_area:
            return 0.5
        att_value = 1 / (1 + np.exp(-self.theta * (area - self.thr_area)))
        return att_value

    def Get_SuppAttMap(self, att_map):
        att = att_map.cpu().numpy()
        his = (cv2.calcHist([att], [0], None, [self.num_bins], [0, 1])).squeeze()
        ind_max = his.argsort()[-1:][0]
        ind_sec = his[ind_max + 1:].argsort()[-1:][0] + ind_max + 1
        thresold = ind_sec / self.num_bins
        bw_map = (np.where(att > thresold, 1, 0).squeeze(0) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(bw_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            if np.any(hierarchy[0, i, 0:2] != [-1, -1]) and np.all(hierarchy[0, i, -2:-1] == [-1, -1]):
                supp_mask = np.zeros_like(bw_map)
                if self.training:
                    if self.gate.sample() == 1:
                        cv2.drawContours(supp_mask, contours, i, 1, -1)
                        con_area = np.sqrt(cv2.contourArea(contours[i]))
                        if con_area > self.min_area:
                            value = self.get_att_value(con_area)
                            supp_mask = supp_mask.astype(np.bool)
                            att_map[0, supp_mask] = torch.pow(att_map[0, supp_mask], 1 / value)

                else:
                    cv2.drawContours(supp_mask, contours, i, 1, -1)
                    con_area = np.sqrt(cv2.contourArea(contours[i]))
                    if con_area > self.min_area:
                        value = self.get_att_value(con_area)
                        supp_mask = supp_mask.astype(np.bool)
                        att_map[0, supp_mask] = torch.pow(att_map[0, supp_mask], 1 / value)

        return att_map

    def forward(self, att_map):
        assert len(att_map.size()) == 4
        new_map = []
        with torch.no_grad():
            for att in att_map:
                long_side = max(att.size())
                if self.training:
                    scale = 16
                else:
                    scale = 32
                self.bin_size = long_side // scale if long_side // scale > 1 else 1
                self.thr_area = self.bin_size
                self.min_area = self.thr_area / 100 if self.thr_area / 100 > 1 else 1
                supp_map = self.Get_SuppAttMap(att)

                new_map.append(supp_map.view(1, supp_map.size()[0],
                                             supp_map.size()[1],
                                             supp_map.size()[2]))
            new_map = torch.cat(new_map, dim=0)

        return new_map


if __name__ == '__main__':
    # sp_m = torch.randn((2, 1, 10, 10))
    # sp_m = torch.Tensor([[0, 0, 0, 0, 0, 0, 1, 1],
    #                      [0, 0, 0, 0, 0, 0, 1, 1],
    #                      [1, 1, 0, 0, 0, 0, 0, 0],
    #                      [1, 1, 0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 1, 1, 0, 0, 0],
    #                      [0, 0, 0, 1, 1, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0, 1, 1],
    #                      [0, 0, 0, 0, 0, 0, 1, 1],
    #                      ])
    # sp_m = sp_m.unsqueeze(dim=0).unsqueeze(dim=0)

    # SPblock = SSPAttBlock()
    # s_n = SPblock(sp_m)

    sc_m = torch.randn((2, 256, 1, 1))
    SCblock = SCAttBlock()
    sc_n = SCblock(sc_m)
