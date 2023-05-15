"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
FLAG_MULTI_LOSS = False
FLAG_EDGE_NO_BG = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from .loss import SoftDiceLoss, FocalTverskyLoss, ACLoss, class2one_hot, one_hot2dist
from .boundary_loss import SurfaceLoss
from . import DualTaskLoss


class JointEdgeDiceLoss(nn.Module):
    def __init__(self, classes, reduction='mean', region_loss='dice', ignore_index=None,
                 norm=False, upper_bound=1.0, mode='train', edge_weight=1,
                 dice_weight=1, edge_type='binary_inner', edge_att_weight=0, seg_grad_weight=0,
                 ac_len_weight=0, ac_region_in_weight=0, ac_region_out_weight=0,
                 boundary_weight=0):
        super(JointEdgeDiceLoss, self).__init__()
        self.num_classes = classes
        # if mode == 'train':
        #     self.seg_loss = ImageBasedCrossEntropyLoss2d(
        #             classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
        # elif mode == 'val':
        if region_loss == 'dice':
            self.region_loss = SoftDiceLoss(n_classes=self.num_classes, ignore_index=ignore_index)
        elif region_loss == 'tversky':
            self.region_loss = FocalTverskyLoss(self.num_classes, False)
        elif region_loss == 'focal_tversky':
            self.region_loss = FocalTverskyLoss(self.num_classes, True)
        else:
            raise NotImplementedError


        self.edge_weight = edge_weight
        self.region_weight = dice_weight
        self.edge_att_weight = edge_att_weight
        self.seg_grad_weight = seg_grad_weight
        self.edge_type = edge_type
        self.ac_len_weight = ac_len_weight
        self.ac_region_in_weight = ac_region_in_weight
        self.ac_region_out_weight = ac_region_out_weight
        self.ac = ac_len_weight or ac_region_in_weight or ac_region_out_weight
        self.boundary_weight = boundary_weight

        if edge_att_weight:
            # self.seg_loss = ImageBasedCrossEntropyLoss3d(classes=classes,
            #         reduction=reduction, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
            # if edge_type.startswith('binary'): self.edge_att_loss = F.binary_cross_entropy_with_logits()
            # if edge_type.startswith('semantic'):
            self.edge_att_loss = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)
                # F.cross_entropy(ignore_index=255)
        if seg_grad_weight:
            raise NotImplementedError
            self.grad_loss = DualTaskLoss(num_classes=classes)

        if self.ac:
            self.ac_loss = ACLoss(n_classes=classes, weight_length=ac_len_weight,
                                  weight_region_in=ac_region_in_weight, weight_region_out=ac_region_out_weight)

        if self.boundary_weight:
            self.boundary_loss = SurfaceLoss(n_classes=classes)

        self.reduction = reduction

    def bce3d(self, input, target):
        n, c, d, h, w = input.size() # c==1 for binary edges, ==n_classes for semantic ones

        # print(input.size(), target.size())
        # target_trans = target_t.clone()

        if self.edge_type.startswith('binary'):
            log_p = input.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(n, -1)
            target_t = target.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(n, -1)
            pos_index = (target_t ==1)
            neg_index = (target_t ==0)
            ignore_index = (target_t >1)

            # target_trans[pos_index] = 1
            # target_trans[neg_index] = 0

            pos_index = pos_index.data.cpu().numpy().astype(bool)
            neg_index = neg_index.data.cpu().numpy().astype(bool)
            ignore_index=ignore_index.data.cpu().numpy().astype(bool)

            # weight = torch.Tensor(log_p.size()).fill_(0)
            # weight = weight.numpy()
            weight = np.zeros(log_p.size())
            pos_num = pos_index.sum()
            neg_num = neg_index.sum()
            sum_num = pos_num + neg_num
            weight[pos_index] = neg_num*1.0 / sum_num
            weight[neg_index] = pos_num*1.0 / sum_num

            weight[ignore_index] = 0

            weight = torch.from_numpy(weight).float()
            weight = weight.cuda()
            loss = F.binary_cross_entropy_with_logits(log_p, target_t.float(), weight, reduction=self.reduction)

        elif self.edge_type.startswith('semantic'):
            if FLAG_MULTI_LOSS:
                target = target.squeeze(dim=1)
                weight = self.class_weight_1hot(target).reshape(self.num_classes - int(FLAG_EDGE_NO_BG), 1, 1, 1)
                target = target.type(torch.float32)
                # if edge is not 1hot
                # target = class2one_hot(target, self.num_classes - int(FLAG_EDGE_NO_BG)).type(torch.float32)

                # print(input.size(), target.size(), weight.size()) # torch.Size([2, 8, 144, 144, 144]) torch.Size([8])

                # weight_sum = target.sum((1, 2, 3, 4)).float()  # BS
                # edge_weight = weight_sum / float(target.size()[2] * target.size()[3] * target.size()[4])
                # nonedge_weight = 1.0 - edge_weight
                # one_sigmoid_out = torch.sigmoid(input)
                # zero_sigmoid_out = 1.0 - one_sigmoid_out
                # # print(input.size()) # torch.Size([2, 8, 144, 144, 144])
                # # print(target.size()) # torch.Size([2, 8, 144, 144, 144])
                # # print(edge_weight.size()) # torch.Size([2])
                # # print(edge_weight, nonedge_weight)
                # # tensor([0.0292, 0.0203], device='cuda:0') tensor([0.9708, 0.9797], device='cuda:0')
                # loss = -nonedge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * target * torch.log(
                #     one_sigmoid_out.clamp(min=1e-10)) - \
                #        edge_weight.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4) * (1 - target) * torch.log(
                #     zero_sigmoid_out.clamp(min=1e-10))
                #
                # loss = loss.mean(dim=0).sum()
                #
                loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
            else:
                target = target.squeeze(dim=1)
                weight = self.class_weight(target)
                loss = F.cross_entropy(input, target, weight, ignore_index=255, reduction=self.reduction)
        else:
            raise NotImplementedError
        return loss

    def class_weight_1hot(self, target):
        all_sum = np.ones(target.size()).sum()
        target_cls = self.num_classes - int(FLAG_EDGE_NO_BG)
        weight = np.zeros(target_cls)
        target_np = target.cpu().numpy().astype(bool)
        for i in range(target_cls):
            single_class_sum = (target_np[:, i+int(FLAG_EDGE_NO_BG), :, :, :]).sum()
            weight[i] = 1 - single_class_sum * 1.0 / all_sum
        weight = torch.from_numpy(weight).float()
        weight = weight.cuda()
        return weight

    def class_weight(self, target):
        # pos_index = (target> 0).data.cpu().numpy().astype(bool)
        # pos_sum = pos_index.sum()
        all_sum = np.ones(target.size()).sum()
        # for CrossEntropyLoss, weight is a Tensor of size C
        target_cls = self.num_classes - int(FLAG_EDGE_NO_BG)
        weight = np.zeros(target_cls)
        for i in range(target_cls):
            single_class_index = (target == (i + int(FLAG_EDGE_NO_BG))).data.cpu().numpy().astype(bool)
            single_class_sum = single_class_index.sum()
            weight[i] = 1 - single_class_sum * 1.0 / all_sum
        weight = torch.from_numpy(weight).float()
        weight = weight.cuda()
        return weight

    def edge_attention(self, input, target, edge):
        n, c, d, h, w = input.size()
        target = target.squeeze(dim=1)
        ignore_filler = torch.ones_like(target) * 255
        # print("edge_att input shape: ", input.size()) # torch.Size([1, 9, 144, 144, 144])
        # print("edge_att target shape: ", target.size()) # torch.Size([1, 144, 144, 144])
        return self.edge_att_loss(input,
                             torch.where(edge.max(1)[0] > 0.8, target, ignore_filler))
    # tensor.max(dim) returns reduced tensor and indices

    def calculate_edge_losses(self, edgein, edgemask, segin, segmask):
        edge_bce = self.edge_weight * self.bce3d(edgein, edgemask) # edgein, edgemask

        edge_ac = 0
        if self.ac:
            edge_ac = self.ac_loss(edgein, edgemask)

        edge_att = 0
        if self.edge_att_weight > 0:
            edge_att = self.edge_att_weight * self.edge_attention(segin, segmask, edgein)
            # segin, edgein
        if self.seg_grad_weight > 0:
            raise NotImplementedError
            # losses['dual_loss'] = self.seg_grad_weight * self.dual_task(segin, segmask)

        return edge_bce + edge_ac + edge_att

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        losses['region_loss'] = self.region_weight * self.region_loss(segin, segmask)
        losses['edge_loss'] = 0

        if not isinstance(edgein, list): edgein = [edgein]
        for single_edge_map in edgein:
            single_edge_loss = self.calculate_edge_losses(single_edge_map, edgemask, segin, segmask)
            losses['edge_loss'] += single_edge_loss

        if self.boundary_weight:
            raise NotImplementedError
            losses['align_loss'] = 0
            # 3 alignments: segin, edgein, edgemask
            gtdist = one_hot2dist(class2one_hot(segmask, self.num_classes))
            # predist = one_hot2dist(class2one_hot(F.softmax(segin.clone().detach(), dim=1).max(1)[1], self.num_classes))

            losses['align_loss'] += self.boundary_loss(edgein[0], gtdist) # edgein, segmask
            # edge_boundary += self.boundary_loss(edgein, predist) # edgein,segin
            losses['align_loss'] *= self.boundary_weight

        return losses

#Img Weighted Loss
class ImageBasedCrossEntropyLoss3d(nn.Module):

    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss3d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.norm = norm
        self.upper_bound = upper_bound
        # self.batch_weights = True

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                          targets[i].unsqueeze(0))
        return loss


#Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

