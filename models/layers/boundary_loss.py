import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from .loss import SoftDiceLoss, GeneralizedDice, class2one_hot, simplex, one_hot

class JointDiceBoundaryLoss(nn.Module):
    def __init__(self, n_classes, dice='softdiceloss', ignore_index=None):
        super(JointDiceBoundaryLoss, self).__init__()
        self.n_classes = n_classes
        if dice == 'softdiceloss':
            self.dice_loss = SoftDiceLoss(n_classes=self.n_classes, ignore_index=ignore_index)
        elif dice == 'generalizeddice':
            self.dice_loss = GeneralizedDice(n_classes=n_classes)
        else:
            raise NotImplementedError

        self.boundary_loss = SurfaceLoss(n_classes=n_classes)

    def forward(self, inputs, targets):
        targets, dist_maps = targets
        losses = {}
        losses['dice_loss'] = self.dice_loss(inputs, targets)
        losses['boundary_loss'] = self.boundary_loss(inputs, dist_maps)
        return losses


class SurfaceLoss():
    def __init__(self, n_classes, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = list(range(n_classes))[1:]

    def __call__(self, input, dist_maps):
        input = F.softmax(input, dim=1)

        assert simplex(input)
        assert not one_hot(dist_maps)

        pc = input[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcdwh,bcdwh->bcdwh", pc, dc)

        loss = multipled.mean()

        return loss

