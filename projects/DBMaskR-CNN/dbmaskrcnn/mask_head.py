# Copyright (c) wondervictor. All Rights Reserved
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY
# from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
import cv2
import numpy as np


def vis_mid(probability_logits, threshold_logits, thresh_binary_logits, pred_instances):
    num_masks = probability_logits.shape[0]
    class_pred = cat([i.pred_classes for i in pred_instances])
    indices = torch.arange(num_masks, device=class_pred.device)
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)
    num_boxes_per_image = [len(i) for i in pred_instances]
    probability_pred = probability_logits[indices, class_pred][:, None]
    probability_pred = probability_pred.split(num_boxes_per_image, dim=0)

    threshold_pred = threshold_logits[indices, class_pred][:, None]
    threshold_pred = threshold_pred.split(num_boxes_per_image, dim=0)

    thresh_binary_pred = thresh_binary_logits[indices, class_pred][:, None]
    thresh_binary_pred = thresh_binary_pred.split(num_boxes_per_image, dim=0)

    for p_pred, t_pred, t_b_pred in zip(probability_pred, threshold_pred, thresh_binary_pred):
        p_pred = p_pred.cpu().data.numpy()[0][0] * 255
        t_pred = t_pred.cpu().data.numpy()[0][0] * 255
        t_b_pred = t_b_pred.cpu().data.numpy()[0][0] * 255

        p_pred = p_pred.astype(np.uint8)
        t_pred = t_pred.astype(np.uint8)
        t_b_pred = t_b_pred.astype(np.uint8)

        cv2.imwrite('p_pred.jpg', p_pred)
        p_pred_map = cv2.applyColorMap(p_pred, cv2.COLORMAP_JET)
        cv2.imwrite('p_pred_map.jpg', p_pred_map)

        cv2.imwrite('thresh.jpg', t_pred)
        t_pred_map = cv2.applyColorMap(t_pred, cv2.COLORMAP_JET)
        cv2.imwrite('color_thresh.jpg', t_pred_map)

        cv2.imwrite('t_b_pred.jpg', t_b_pred)
        t_b_pred_map = cv2.applyColorMap(t_b_pred, cv2.COLORMAP_JET)
        cv2.imwrite('color_t_b_pred.jpg', t_b_pred_map)


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None]
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def db_loss_func(db_logits, gtmasks):
    """
    Args:
        db_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                                    dtype=torch.float32,
                                    device=db_logits.device).reshape(
                                        1, 1, 3, 3).requires_grad_(False)
    db_logits = db_logits.unsqueeze(1)
    db_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    db_targets = db_targets.clamp(min=0)
    db_targets[db_targets > 0.1] = 1
    db_targets[db_targets <= 0.1] = 0

    if db_logits.shape[-1] != db_targets.shape[-1]:
        db_targets = F.interpolate(db_targets,
                                   db_logits.shape[2:],
                                   mode='nearest')

    bce_loss = F.binary_cross_entropy(db_logits, db_targets)
    dice_loss = dice_loss_func(db_logits, db_targets)
    return bce_loss + dice_loss


def db_preserving_mask_loss(probability_logits,
                            threshold_logits,
                            thresh_binary,
                            instances,
                            threshold_on=True,
                            vis_period=0):
    cls_agnostic_mask = probability_logits.size(1) == 1
    total_num_masks = thresh_binary.size(0)
    mask_side_len = probability_logits.size(2)
    assert probability_logits.size(2) == probability_logits.size(
        3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(
                dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor,
            mask_side_len).to(device=probability_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return probability_logits.sum() * 0, threshold_logits.sum(
        ) * 0, thresh_binary.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        probability_logits = probability_logits[:, 0]
        threshold_logits = threshold_logits[:, 0]
        thresh_binary = thresh_binary[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        probability_logits = probability_logits[indices, gt_classes]
        threshold_logits = threshold_logits[indices, gt_classes]
        thresh_binary = thresh_binary[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (thresh_binary > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() /
                         max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0)
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(
        num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = thresh_binary
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    probability_loss = F.binary_cross_entropy(probability_logits,
                                              gt_masks,
                                              reduction="mean")
    if threshold_on:
        threshold_loss = db_loss_func(threshold_logits, gt_masks)
    else:
        threshold_loss = threshold_logits.sum() * 0
    thresh_binary_loss = F.binary_cross_entropy(thresh_binary,
                                                gt_masks,
                                                reduction="mean")
    return probability_loss, threshold_loss, thresh_binary_loss


@ROI_MASK_HEAD_REGISTRY.register()
class DBPreservingHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(DBPreservingHead, self).__init__()

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_db_conv = cfg.MODEL.DB_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1
        self.adaptive = cfg.MODEL.DB_MASK_HEAD.ADAPTIVE
        self.fusion = None
        self.k = 50
        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_final_fusion = Conv2d(conv_dim,
                                        conv_dim,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        bias=not conv_norm,
                                        norm=get_norm(conv_norm, conv_dim),
                                        activation=F.relu)

        self.downsample = Conv2d(conv_dim,
                                 conv_dim,
                                 kernel_size=3,
                                 padding=1,
                                 stride=2,
                                 bias=not conv_norm,
                                 norm=get_norm(conv_norm, conv_dim),
                                 activation=F.relu)
        self.db_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_db_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("db_fcn{}".format(k + 1), conv)
            self.db_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_to_db = Conv2d(conv_dim,
                                 conv_dim,
                                 kernel_size=1,
                                 padding=0,
                                 stride=1,
                                 bias=not conv_norm,
                                 norm=get_norm(conv_norm, conv_dim),
                                 activation=F.relu)

        self.db_to_mask = Conv2d(conv_dim,
                                 conv_dim,
                                 kernel_size=1,
                                 padding=0,
                                 stride=1,
                                 bias=not conv_norm,
                                 norm=get_norm(conv_norm, conv_dim),
                                 activation=F.relu)

        self.mask_deconv = ConvTranspose2d(conv_dim,
                                           conv_dim,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        self.mask_predictor = Conv2d(cur_channels,
                                     num_classes,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

        self.db_deconv = ConvTranspose2d(conv_dim,
                                         conv_dim,
                                         kernel_size=2,
                                         stride=2,
                                         padding=0)
        self.db_predictor = Conv2d(cur_channels,
                                   num_classes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        for layer in self.mask_fcns + self.db_fcns + \
                     [self.mask_deconv, self.db_deconv, self.db_to_mask, self.mask_to_db,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.db_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.db_predictor.bias is not None:
            nn.init.constant_(self.db_predictor.bias, 0)

    def forward(self, mask_features, db_features, instances: List[Instances]):
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)
        # downsample
        db_features = self.downsample(db_features)
        # mask to db fusion
        db_features = db_features + self.mask_to_db(mask_features)
        for layer in self.db_fcns:
            db_features = layer(db_features)
        # db to mask fusion
        mask_features = self.db_to_mask(db_features) + mask_features
        mask_features = self.mask_final_fusion(mask_features)
        # mask prediction
        mask_features = F.relu(self.mask_deconv(mask_features))
        probability_logits = self.mask_predictor(mask_features).sigmoid()
        # db prediction
        db_features = F.relu(self.db_deconv(db_features))
        threshold_logits = self.db_predictor(db_features).sigmoid()

        thresh_binary = self.step_function(probability_logits, threshold_logits)
        if self.training:
            loss_probability, loss_threshold, loss_threshold_binary = db_preserving_mask_loss(
                probability_logits, threshold_logits, thresh_binary, instances)
            return {
                "loss_binary": loss_probability,
                "loss_threshold": loss_threshold,
                "loss_threshold_binary": loss_threshold_binary
            }
        else:
            if self.adaptive:
                mask_rcnn_inference(thresh_binary, instances)
            else:
                mask_rcnn_inference(probability_logits, instances)
            return instances

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


@ROI_MASK_HEAD_REGISTRY.register()
class DBMASKHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(DBMASKHead, self).__init__()

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_db_conv = cfg.MODEL.DB_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1
        self.adaptive = cfg.MODEL.DB_MASK_HEAD.ADAPTIVE
        self.fusion = False
        self.k = 50
        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.db_fcns = []
        if self.fusion:
            cur_channels = input_shape.channels + num_classes
        else:
            cur_channels = input_shape.channels
        for k in range(num_db_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("db_fcn{}".format(k + 1), conv)
            self.db_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_deconv = ConvTranspose2d(conv_dim,
                                           conv_dim,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        self.mask_predictor = Conv2d(cur_channels,
                                     num_classes,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

        self.threshold_predictor = Conv2d(cur_channels,
                                          num_classes,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)

        for layer in self.mask_fcns + self.db_fcns + [self.mask_deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.threshold_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.threshold_predictor.bias is not None:
            nn.init.constant_(self.threshold_predictor.bias, 0)

    def forward(self, mask_features, threshold_features, instances: List[Instances]):
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)
        # probability
        mask_features = F.relu(self.mask_deconv(mask_features))
        probability_logits = self.mask_predictor(mask_features).sigmoid()

        if self.fusion:
            threshold_features = torch.cat((threshold_features, probability_logits), 1)
        for layer in self.db_fcns:
            threshold_features = layer(threshold_features)
        # threshold prediction
        threshold_logits = self.threshold_predictor(threshold_features).sigmoid()

        thresh_binary = self.step_function(probability_logits, threshold_logits)
        # vis_mid(probability_logits, threshold_logits, thresh_binary, instances)
        if self.training:
            loss_probability, loss_threshold, loss_threshold_binary = db_preserving_mask_loss(
                probability_logits, threshold_logits, thresh_binary, instances)
            return {
                "loss_binary": loss_probability,
                "loss_threshold": loss_threshold,
                "loss_threshold_binary": loss_threshold_binary
            }
        else:
            if self.adaptive:
                mask_rcnn_inference(thresh_binary, instances)
            else:
                mask_rcnn_inference(probability_logits, instances)
            return instances

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
