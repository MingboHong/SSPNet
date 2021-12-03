import numpy as np
import cv2
import torch
from mmdet.models.builder import build_loss
from mmdet.core import multi_apply_single, multi_apply


class Heatmap:
    def __init__(self, fpn_lvl=4,
                 loss_att=dict(
                     type='CrossEntropyLoss',
                     bce_use_sigmoid=True,
                     loss_weight=1.0)):
        self.nb_downsample = 2
        self.fpn_lvl = fpn_lvl
        self.lamda = 0.1
        self.loss_att = build_loss(loss_att)

    @staticmethod
    def get_bbox_mask(area, lvl):

        min_area = 2 ** (lvl + 2)
        max_area = 2 ** (lvl + 5)
        if min_area < area < max_area:
            return 1
        elif lvl == 0 and area < min_area:  # scale <4  all 1
            return 1
        elif lvl == 3 and area > min_area:  # scale> range of top  all 1
            return 1
        else:
            return -1

    @staticmethod
    def seg_loss(pred, target, mask):

        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        target[target > 0] = 1

        pred = pred * mask
        target = target * mask

        a = torch.sum(pred * target, 1) + 1  # + 1  # 0.001
        b = torch.sum(pred * pred, 1) + 1  # + 1
        c = torch.sum(target * target, 1) + 1  # + 1
        d = (2 * a) / (b + c)
        loss = 1 * (1 - d)
        return loss

    def reg_loss(self, pred, target, weight):
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        weight = weight.contiguous().view(weight.size()[0], -1)

        mask = weight.clone()
        mask[mask > 0] = 1

        pred = pred * mask
        target = target * mask

        num_total_samples = len(mask[mask > 0])
        num_total_samples = num_total_samples if num_total_samples > 0 else None
        loss = self.loss_att(pred, target, weight, avg_factor=num_total_samples)

        return 0.01 * loss

    @staticmethod
    def reg_mask(pred_att, gt_att):
        pos_num = int(np.sum(gt_att > 0.0))

        if pos_num == 0:
            total_num = gt_att.shape[0] * gt_att.shape[1] * gt_att.shape[2]
            neg_num = int(0.05 * total_num)
            neg_score = pred_att[gt_att == 0.0]
            neg_score_sorted = np.sort(-neg_score)
            threshold = -neg_score_sorted[neg_num - 1]
            selected_mask = (pred_att >= threshold) | (gt_att < 0.0)
            selected_mask = selected_mask.reshape(1, gt_att.shape[1], gt_att.shape[2]).astype('float32')
            selected_mask[(gt_att < 0.0)] = 3
            return selected_mask

        neg_num = int(np.sum(gt_att == 0.0))
        neg_num = int(min(pos_num * 3, neg_num))

        neg_score = pred_att[gt_att == 0.0]

        # 将负样本得分从高到低排序
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]

        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((pred_att >= threshold) | (gt_att > 0.0) | (gt_att < 0.0))
        selected_mask = selected_mask.reshape(1, selected_mask.shape[1], selected_mask.shape[2]).astype('float32')
        selected_mask[(gt_att < 0.0)] = 3
        return selected_mask

    @staticmethod
    def seg_mask(gt_att):
        pos_num = int(np.sum(gt_att > 0.0))

        if pos_num == 0:
            # selected_mask = (gt_att < 0.0)
            selected_mask = gt_att.copy() * 0
            return selected_mask

        selected_mask = (gt_att > 0.0)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[1], selected_mask.shape[2]).astype('float32')

        return selected_mask

    def mask_batch(self, pred_att, gt_att):

        pred_att = pred_att.data.cpu().numpy()
        gt_att = gt_att.data.cpu().numpy()

        selected_reg = []
        selected_seg = []
        for i in range(pred_att.shape[0]):
            selected_reg.append(self.reg_mask(pred_att[i, :, :, :], gt_att[i, :, :, :]))
            selected_seg.append(self.seg_mask(gt_att[i, :, :, :]))

        selected_reg = np.concatenate(selected_reg, 0)
        selected_seg = np.concatenate(selected_seg, 0)
        selected_reg = torch.from_numpy(selected_reg).float()
        selected_seg = torch.from_numpy(selected_seg).float()

        return selected_reg, selected_seg

    def loss_single(self, pred, gt):

        selected_reg_masks, selected_seg_masks = self.mask_batch(pred, gt)
        selected_reg_masks = selected_reg_masks.to(pred.device)
        selected_seg_masks = selected_seg_masks.to(pred.device)
        gt[gt < 0] = 0
        loss_reg = self.reg_loss(pred, gt, selected_reg_masks)
        loss_seg = self.seg_loss(pred, gt, selected_seg_masks)

        return loss_reg, loss_seg

    def loss(self, reg_pred, reg_gt):

        losses_reg, losses_seg = multi_apply(self.loss_single, reg_pred, reg_gt)
        return dict(loss_reg=losses_reg, loss_seg=losses_seg)

    def target_single(self, anns, lvl, img_h, img_w):
        gt_mp = np.zeros((img_h, img_w))
        for ann in anns:
            x1, y1, x2, y2 = ann

            l = np.int(x1)
            t = np.int(y1)

            r = int(np.clip(np.ceil(x2), a_min=0, a_max=img_w - 1))
            d = int(np.clip(np.ceil(y2), a_min=0, a_max=img_h - 1))

            w = r - l
            h = d - t

            value = self.get_bbox_mask(np.sqrt(w * h), lvl)
            gt_mp[t:d, l:r] = value

        gt_mp = cv2.resize(gt_mp,
                           (img_w // (2 ** (lvl + self.nb_downsample)), img_h // (2 ** (lvl + self.nb_downsample))))
        gt_mp = gt_mp[np.newaxis, np.newaxis, :, :]
        return torch.from_numpy(gt_mp)

    def target(self, pred_att, anns):

        self.fpn_lvl = len(pred_att)

        batch_size, feats_c, feats_height, feats_width = pred_att[0].shape

        anns_t = [[anns[i].detach().cpu().numpy() for i in range(batch_size)]] * self.fpn_lvl

        lvl = np.arange(self.fpn_lvl)[:, np.newaxis].repeat(4, axis=-1)

        img_height = feats_height * (2 ** self.nb_downsample)
        img_width = feats_width * (2 ** self.nb_downsample)

        mask_target = []
        for i in range(self.fpn_lvl):
            lvl_target = map(self.target_single, anns_t[i], lvl[i], np.full_like(lvl[i], img_height),
                             np.full_like(lvl[i], img_width))
            lvl_target = list(lvl_target)
            mask_target.append(torch.cat(lvl_target).to(device=pred_att[i].device))

        return tuple(mask_target)
