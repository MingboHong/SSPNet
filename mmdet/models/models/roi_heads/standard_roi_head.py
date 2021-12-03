import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import numpy as np
import scipy


@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, **kwargs):
        super(StandardRoIHead, self).__init__(**kwargs)
        self.area_history = [[], [], [], [], []]
        self.fpn_range = {'level_0': [], 'level_1': [], 'level_2': [], 'level_3': [],
                          'level_4': []}  # [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.fpn_lvls = 5
        self.iter = 0
        self.topk = 3

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def update_area_range_iter(self):
        for i in range(self.fpn_lvls):
            if len(self.area_history[i]) > 100:
                if isinstance(self.area_history[i], list):
                    temp_area = torch.cat(self.area_history[i], 1)
                else:
                    temp_area = self.area_history[i]
                sample_size = temp_area.numel()

                z_score = scipy.stats.norm.isf(0.025)
                mean = temp_area.mean().item()
                std = temp_area.std().item()
                margin_error = (z_score * std) / np.sqrt(sample_size)
                low_area = (mean - margin_error) if (mean - margin_error) > 0 else 0
                high_area = mean + margin_error
                if len(self.fpn_range['level_{}'.format(i)]) == 0:
                    self.fpn_range['level_{}'.format(i)] = [low_area, high_area]
                else:
                    self.fpn_range['level_{}'.format(i)] = [(self.fpn_range['level_{}'.format(i)][0] + low_area) / 2,
                                                            (self.fpn_range['level_{}'.format(i)][1] + high_area) / 2]
                print('the {} level range is {}'.format(i, self.fpn_range['level_{}'.format(i)]))
                self.area_history[i] = []
        with open('./result/fpn_range.txt', 'a+')as f:
            f.write('iter:%d :' % self.iter+str(self.fpn_range)+'\n')

    def update_area_range(self, lvl):
        import scipy
        if isinstance(self.area_history[lvl], list):
            temp_area = torch.cat(self.area_history[lvl], 1)
        else:
            temp_area = self.area_history[lvl]
        sample_size = temp_area.numel()
        z_score = scipy.stats.norm.isf(0.025)
        mean = temp_area.mean().item()
        std = temp_area.std().item()
        margin_error = (z_score * std) / np.sqrt(sample_size)
        low_area = (mean - margin_error) if (mean - margin_error) > 0 else 0
        high_area = mean + margin_error
        if len(self.fpn_range['level_{}'.format(lvl)]) == 0:
            self.fpn_range['level_{}'.format(lvl)] = [low_area, high_area]
        else:
            self.fpn_range['level_{}'.format(lvl)] = 0.5 * self.fpn_range['level_{}'.format(lvl)] + 0.5 * [low_area,
                                                                                                           high_area]
        print('the {} level range is {}'.format(lvl, self.fpn_range['level_{}'.format(lvl)]))
        self.area_history[lvl] = []

    def record_area_range(self, proposal_list, lvl_id, assign_result):
        gt_num = assign_result.num_gts
        for i in range(self.fpn_lvls):
            # get the current level's index
            lvl_inds = torch.where(lvl_id[0] == i)

            # get the iou corresponding to the current level
            iou = assign_result.max_overlaps[lvl_inds]

            # get the sample's id corresponding to the current level
            gt_lvl_inds = assign_result.gt_inds[lvl_inds]
            topk_inds = []

            for n in range(1, gt_num + 1):
                gt_inds = torch.where(gt_lvl_inds == n)
                gt_ious = iou[gt_inds]
                if len(gt_inds) > 0:
                    sort_iou, inds = gt_ious.sort(descending=True)
                    topk = self.topk if self.topk > len(gt_inds) else len(gt_inds)
                    topk_inds.append(gt_inds[0][inds[:topk]])

            # sort_iou, inds = iou.sort(descending=True)
            #
            # # topk = (len((sort_iou > 0.0).nonzero()) - 1) if (len(
            # #     (sort_iou > 0.0).nonzero()) - 1) < self.topk else self.topk
            #
            # topk = len((sort_iou > 0.5).nonzero(as_tuple=False)) - 1 if len(
            #     (sort_iou > 0.5).nonzero(as_tuple=False)) > 0 else 0

            keep = lvl_inds[0][torch.cat(topk_inds, 0)]
            if len(keep) > 0:
                area = (proposal_list[0][keep][:, 2] - proposal_list[0][keep][:, 0]) * (
                        proposal_list[0][keep][:, 3] - proposal_list[0][keep][:, 1])
                if len(self.area_history[i]) > 0:
                    self.area_history[i] = torch.cat([torch.sqrt(area), self.area_history[i]], 0)
                else:
                    self.area_history[i] = area

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'],)
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            #lvl_id = [proposal_list[i][1] for i in range(len(proposal_list))]
           # proposal_list = [proposal_list[i][0] for i in range(len(proposal_list))]

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
               # self.record_area_range(proposal_list, lvl_id, assign_result)
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        # if self.iter % 100 == 0:
        #     self.update_area_range_iter()
        # # for i in range(self.fpn_lvls):
        # #     if len(self.area_history[i]) > 1000:
        # # @        self.update_area_range(i)
        # self.iter += 1
        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
