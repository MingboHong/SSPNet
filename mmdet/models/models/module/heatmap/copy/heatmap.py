import numpy as np
import cv2
import torch
from mmdet.models.builder import build_loss
from mmdet.core import multi_apply_single, multi_apply
import math


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

    def get_bbox_mask(self, area, lvl):

        min_area = 2 ** (lvl + 2)
        max_area = 2 ** (lvl + 6)
        if min_area < area < max_area:
            return 1
        elif lvl == 0 and area < min_area:  # scale <4  all 1
            return 1
        elif lvl == 3 and area > min_area:  # scale> range of top  all 1
            return 1
        else:
            return 0

    def seg_loss(self, pred, target, mask):

        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        pred = pred * mask
        target[target > 0] = 1
        target = target * mask

        a = torch.sum(pred * target, 1) + 1  # + 1  # 0.001
        b = torch.sum(pred * pred, 1) + 1  # + 1
        c = torch.sum(target * target, 1) + 1  # + 1
        d = (2 * a) / (b + c)
        loss = 0.1 * (1 - d)
        return loss

    def reg_loss(self, pred, target, mask):

        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        pred = pred * mask
        target = target * mask
        target_mask = target.clone()
        target_mask[target_mask > 0] = 1

        num_total_samples = len(target_mask[target_mask > 0])
        num_total_samples = num_total_samples if num_total_samples > 0 else None
        loss = self.loss_att(pred, target, target_mask, avg_factor=num_total_samples)

        return 0.02 * loss

    def ohem_single(self, pred_att, gt_att):
        pos_num = int(np.sum(gt_att > 0.0))

        if pos_num == 0:
            selected_mask = gt_att.copy()
            # selected_mask = gt_text.copy() * 0 # may be not good
            total_num = gt_att.shape[0] * gt_att.shape[1] * gt_att.shape[2]
            neg_num = int(0.05 * total_num)
            mask_range = np.arange(0, gt_att.shape[0] * gt_att.shape[1] * gt_att.shape[2])
            neg_index = np.random.choice(mask_range, neg_num, replace=False)
            selected_mask = selected_mask.reshape(-1)
            selected_mask[neg_index] = 1
            selected_mask = selected_mask.reshape(1, gt_att.shape[1], gt_att.shape[2]).astype('float32')
            return selected_mask
        # neg_num = int(np.sum(gt_att == 0.0))
        # neg_num = int(min(pos_num * 3, neg_num))
        # selected_mask = gt_att.copy()
        # fg_masks = (gt_att != 0.0).reshape(-1, )
        # mask_range = np.arange(1, gt_att.shape[0] * gt_att.shape[1] * gt_att.shape[2] + 1)
        # mask_range[fg_masks] = 0
        # gt_index = list(set(mask_range))
        # gt_index.remove(0)
        # neg_index = np.random.choice(gt_index, neg_num, replace=False)
        # selected_mask = selected_mask.reshape(-1)
        # selected_mask[neg_index - 1] = 1
        # selected_mask[fg_masks] = 1



        neg_num = int(np.sum(gt_att == 0.0))
        neg_num = int(min(pos_num * 3, neg_num))

        neg_score = pred_att[gt_att == 0.0]
        # 将负样本得分从高到低排序
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        # threshold = threshold if threshold > 0.3 else 0.3
        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((pred_att >= threshold) | (gt_att > 0.0))
        selected_mask = selected_mask.reshape(1, selected_mask.shape[1], selected_mask.shape[2]).astype('float32')

        return selected_mask

    def ohem_batch(self, pred_att, gt_att):
        pred_att = pred_att.data.cpu().numpy()
        gt_att = gt_att.data.cpu().numpy()

        selected_masks = []
        for i in range(pred_att.shape[0]):
            selected_masks.append(self.ohem_single(pred_att[i, :, :, :], gt_att[i, :, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks

    def loss_single(self, reg_pred, reg_gt):

        selected_reg_masks = self.ohem_batch(reg_pred, reg_gt)
        selected_reg_masks = selected_reg_masks.to(reg_pred.device)
        loss_reg = self.reg_loss(reg_pred, reg_gt, selected_reg_masks)
        loss_seg = self.seg_loss(reg_pred, reg_gt, selected_reg_masks)

        return loss_reg, loss_seg

    def loss(self, reg_pred, reg_gt):

        losses_reg, losses_seg = multi_apply(self.loss_single, reg_pred, reg_gt)
        return dict(loss_reg=losses_reg, loss_seg=losses_seg)

    def target(self, feats_height, feats_width, anns):
        if isinstance(anns, torch.Tensor):
            anns_t = anns.detach().cpu().numpy()
        else:
            anns_t = anns
        gt_map = []

        img_height = feats_height * (2 ** self.nb_downsample)
        img_width = feats_width * (2 ** self.nb_downsample)

        for lvl in range(self.fpn_lvl):
            gt_map.append(np.zeros([img_height, img_width],
                                   dtype=np.float32))
        for lvl in range(self.fpn_lvl):
            if len(anns_t) > 0:
                for ann in anns_t:
                    x1, y1, x2, y2 = ann
                    x_beg = int(x1)
                    y_beg = int(y1)
                    x_end = np.clip(int(x2), a_min=0, a_max=img_width - 1)
                    y_end = np.clip(int(y2), a_min=0, a_max=img_height - 1)
                    # x_end = np.clip(math.ceil(x2), a_min=0, a_max=img_width - 1)
                    # y_end = np.clip(math.ceil(y2), a_min=0, a_max=img_height - 1)
                    w = x2 - x1
                    h = y2 - y1
                    value = self.get_bbox_mask(np.sqrt(w * h), lvl)
                    gt_map[lvl][y_beg:y_end, x_beg:x_end] = value

                gt_map[lvl] = cv2.resize(gt_map[lvl], (int(feats_width / (2 ** lvl)), int(feats_height / (2 ** lvl))))
                gt_map[lvl] = anns.new_tensor(gt_map[lvl]).view(1, 1,
                                                                gt_map[lvl].shape[0],
                                                                gt_map[lvl].shape[1])

        return tuple(gt_map)

    # att = selected_mask*255
    # att = att.astype(np.uint8).transpose(1, 2, 0)
    # cv2.imshow("img", att)
    # cv2.waitKey(0)

    # selected_mask = gt_att.copy()
    # fg_masks = (gt_att != 0.0).reshape(-1, )
    # mask_range = np.arange(1, gt_att.shape[0] * gt_att.shape[1] * gt_att.shape[2] + 1)
    # mask_range[fg_masks] = 0
    # gt_index = list(set(mask_range))
    # gt_index.remove(0)
    # neg_index = np.random.choice(gt_index, neg_num, replace=False)
    # selected_mask = selected_mask.reshape(-1)
    # selected_mask[neg_index - 1] = 1
    # selected_mask[fg_masks] = 1
    # neg_score = pred_att[gt_att == 0.0]
    # selected_mask = (selected_mask | (gt_att > 0.0))
