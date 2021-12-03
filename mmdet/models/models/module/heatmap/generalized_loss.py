import numpy as np
import torch
from mmdet.core import multi_apply
from mmdet.models.builder import build_loss
import cv2


class AHLoss:
    def __init__(self, fpn_lvl=4,
                 loss_att=dict(
                     type='CrossEntropyLoss',
                     bce_use_sigmoid=True,
                     loss_weight=1.0),
                 anchor_scales=None,
                 ObjBgW=3.0,
                 alpha=1.0,
                 beta=0.01,
                 ratio=3.0):
        self.alpha = alpha
        self.beta = beta
        if anchor_scales is None:
            anchor_scales = [1.5, 8.0]
        self.ratio = ratio
        self.ObjBgW = ObjBgW
        self.nb_downsample = 2
        self.fpn_lvl = fpn_lvl
        self.loss_att = build_loss(loss_att)
        self.anchor_scales = anchor_scales

    def get_bbox_mask(self, area, lvl):

        min_area = 0.5 * min(self.anchor_scales)
        max_area = 2 * max(self.anchor_scales)

        if min_area < area < max_area:
            return 1
        elif lvl == 0 and area < min_area:  # scale <4  all 1
            return 1
        elif lvl == 3 and area > min_area:  # scale> range of top  all 1
            return 1
        else:
            return -1

    # @staticmethod
    def seg_loss(self, pred, target, mask):

        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(pred.size()[0], -1)
        mask = mask.contiguous().view(pred.size()[0], -1)

        gt = target.clone()
        gt[gt > 0] = 1
        pred = pred * mask
        gt = gt * mask

        a = torch.sum(pred * gt, 1) + 1  # + 1  # 0.001
        b = torch.sum(pred * pred, 1) + 1  # + 1
        c = torch.sum(gt * gt, 1) + 1  # + 1
        d = (2 * a) / (b + c)
        return self.alpha * (1 - d)

    def reg_loss(self, pred, target, weight):
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(pred.size()[0], -1)
        weight = weight.contiguous().view(pred.size()[0], -1)

        mask = weight.clone()
        mask[mask > 0] = 1

        pred = pred * mask
        target = target * mask

        num_total_samples = len(mask[mask > 0])
        num_total_samples = num_total_samples if num_total_samples > 0 else None
        loss = self.loss_att(pred, target, weight, avg_factor=num_total_samples)

        return self.beta * loss

    def reg_mask(self, pred_att, gt_att):
        b, h, w = pred_att.shape[0], pred_att.shape[-2], pred_att.shape[-1]

        att = np.zeros([b, h, w])

        for i in range(len(gt_att)):
            att += gt_att[i]
        att[att > 1] = 1
        pos_num = int(np.sum(att > 0.0))

        if pos_num == 0:
            total_num = att.shape[1] * att.shape[2]
            neg_num = int(0.05 * total_num)
            neg_score = pred_att[att == 0.0]
            neg_score_sorted = np.sort(-neg_score)
            threshold = -neg_score_sorted[neg_num - 1]
            selected_mask = (pred_att >= threshold) | (att < 0.0)
            selected_mask = selected_mask.reshape(1, att.shape[1], att.shape[2]).astype('float32')
            selected_mask[(att < 0.0)] = self.ObjBgW
            return selected_mask

        neg_num = int(np.sum(att == 0.0))
        neg_num = int(min(pos_num * self.ratio, neg_num))

        neg_score = pred_att[att == 0.0]

        # 将负样本得分从高到低排序
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]

        # 选出 得分高的 负样本 和正样本 的 mask
        selected_mask = ((pred_att >= threshold) | (att > 0.0) | (att < 0.0))
        selected_mask = selected_mask.reshape(1, selected_mask.shape[1], selected_mask.shape[2]).astype('float32')
        selected_mask[(att < 0.0)] = self.ObjBgW
        return selected_mask

    @staticmethod
    def seg_mask(gt_att):
        gt_nums = len(gt_att)
        masks = []

        for i in range(gt_nums):
            pos_num = int(np.sum(gt_att[i] > 0.0))

            if pos_num == 0:
                selected_mask = gt_att[i].copy() * 0
            else:
                selected_mask = (gt_att[i] > 0.0).astype(
                    'float32')
            masks.append(selected_mask)

        return masks

    def mask_batch(self, pred_att, gt_att):
        pred_att = pred_att.data.cpu().numpy()
        gt_att = gt_att.unsqueeze(0).data.cpu().numpy()
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
        gt_nums = len(gt)

        valid_nums = 0
        loss_seg = 0
        for i in range(gt_nums):
            gt[i][gt[i] < 0] = 0
            loss_seg += self.seg_loss(pred, gt[i], selected_seg_masks[i])
            if torch.sum(gt[i] > 0) > 0:
                valid_nums += 1
        loss_seg = loss_seg / valid_nums if valid_nums > 0 else loss_seg

        reg_gt = torch.zeros_like(gt[0])
        for i in range(gt_nums):
            reg_gt += gt[i]
        reg_gt[reg_gt > 1] = 1
        loss_reg = self.reg_loss(pred, reg_gt, selected_reg_masks)

        return loss_reg, loss_seg

    def loss(self, reg_pred, reg_gt):

        losses_reg, losses_seg = multi_apply(self.loss_single, reg_pred, reg_gt)
        return dict(loss_reg=losses_reg, loss_seg=losses_seg)

    def target(self, feats_height, feats_width, anns):

        gt_map = [[] for _ in range(self.fpn_lvl)]
        num_gts = len(anns)

        img_height = feats_height * (2 ** self.nb_downsample)
        img_width = feats_width * (2 ** self.nb_downsample)

        if num_gts > 0:
            for lvl in range(self.fpn_lvl):
                for gt in range(max(num_gts, 1)):
                    gt_map[lvl].append(
                        np.zeros(
                            [img_height, img_width],
                            dtype=np.float32))

        for lvl in range(self.fpn_lvl):
            if num_gts > 0:
                for i, ann in enumerate(anns):
                    x1, y1, x2, y2 = ann.detach().cpu().numpy()  # / (2 ** (self.nb_downsample + lvl))

                    w = x2 - x1
                    h = y2 - y1

                    value = self.get_bbox_mask(np.sqrt(w * h) / (2 ** (self.nb_downsample + lvl)), lvl)
                    x1, y1, x2, y2 = list(map(round, [x1, y1, x2, y2]))

                    x1 = np.clip(x1, a_min=0, a_max=img_width - 1)
                    y1 = np.clip(y1, a_min=0, a_max=img_height - 1)
                    x2 = np.clip(x2, a_min=0, a_max=img_width - 1)
                    y2 = np.clip(y2, a_min=0, a_max=img_height - 1)

                    gt_map[lvl][i][y1:y2, x1:x2] = value

                    gt_map[lvl][i] = anns.new_tensor(cv2.resize(gt_map[lvl][i],
                                                                (feats_width // (2 ** lvl), feats_height // (2 ** lvl))
                                                                ), dtype=torch.float32)

                gt_map[lvl] = torch.stack(gt_map[lvl])
        return tuple(gt_map)


"""
   def target(self, feats_height, feats_width, anns):

        gt_map = [[] for _ in range(self.fpn_lvl)]
        num_gts = len(anns)

        if num_gts > 0:
            for lvl in range(self.fpn_lvl):
                for gt in range(max(num_gts, 1)):
                    gt_map[lvl].append(
                        anns.new_zeros([feats_height // (2 ** lvl), feats_width // (2 ** lvl)], dtype=torch.float32))

        for lvl in range(self.fpn_lvl):
            if num_gts > 0:
                for id, ann in enumerate(anns):
                    x1, y1, x2, y2 = ann / (2 ** (self.nb_downsample + lvl))
                    w = x2 - x1
                    h = y2 - y1

                    value = self.get_bbox_mask(torch.sqrt(w * h), lvl)
                    x1, y1, x2, y2 = list(map(torch.round, [x1, y1, x2, y2]))

                    x1 = x1.long().clamp(min=0, max=feats_width - 1)
                    y1 = y1.long().clamp(min=0, max=feats_height - 1)
                    x2 = x2.long().clamp(min=0, max=feats_width - 1)
                    y2 = y2.long().clamp(min=0, max=feats_height - 1)

                    if x1 == x2:
                        x2 += 1
                    if y1 == y2:
                        y2 += 1

                    gt_map[lvl][id][y1:y2, x1:x2] = value
                gt_map[lvl] = torch.stack(gt_map[lvl])
        return tuple(gt_map)
"""
"""

 def target(self, feats_height, feats_width, anns):

        gt_map = [[] for _ in range(self.fpn_lvl)]
        num_gts = len(anns)

        img_height = feats_height * (2 ** self.nb_downsample)
        img_width = feats_width * (2 ** self.nb_downsample)

        if num_gts > 0:
            for lvl in range(self.fpn_lvl):
                for gt in range(max(num_gts, 1)):
                    gt_map[lvl].append(
                        np.zeros(
                            [img_height, img_width],
                            dtype=np.float32))

        for lvl in range(self.fpn_lvl):
            if num_gts > 0:
                for i, ann in enumerate(anns):
                    x1, y1, x2, y2 = ann.detach().cpu().numpy()  # / (2 ** (self.nb_downsample + lvl))

                    w = x2 - x1
                    h = y2 - y1

                    value = self.get_bbox_mask(np.sqrt(w * h) / (2 ** (self.nb_downsample + lvl)), lvl)
                    x1, y1, x2, y2 = list(map(round, [x1, y1, x2, y2]))

                    x1 = np.clip(x1, a_min=0, a_max=img_width - 1)
                    y1 = np.clip(y1, a_min=0, a_max=img_height - 1)
                    x2 = np.clip(x2, a_min=0, a_max=img_width - 1)
                    y2 = np.clip(y2, a_min=0, a_max=img_height - 1)

                    gt_map[lvl][i][y1:y2, x1:x2] = value

                    gt_map[lvl][i] = anns.new_tensor(cv2.resize(gt_map[lvl][i],
                                                                (feats_width // (2 ** lvl), feats_height // (2 ** lvl))
                                                                ), dtype=torch.float32)

                gt_map[lvl] = torch.stack(gt_map[lvl])

"""