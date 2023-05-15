"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code adapted from:
# https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
#
# MIT License
#
# Copyright (c) 2016 Eric Jang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def perturbate_input_(input, n_elements=200):
    N, C, H, W = input.shape
    assert N == 1
    c_ = np.random.random_integers(0, C - 1, n_elements)
    h_ = np.random.random_integers(0, H - 1, n_elements)
    w_ = np.random.random_integers(0, W - 1, n_elements)
    for c_idx in c_:
        for h_idx in h_:
            for w_idx in w_:
                input[0, c_idx, h_idx, w_idx] = 1
    return input

def _sample_gumbel(shape, eps=1e-10):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).cuda()
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    assert logits.dim() == 3
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, 1)


def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes).cuda()
    return y[labels].permute(0,3,1,2)

class DualTaskLoss(nn.Module):
    def __init__(self, num_classes=15, cuda=False):
        super(DualTaskLoss, self).__init__()
        self._cuda = cuda
        self.num_classes = num_classes
        # return

    def forward(self, input_logits, gts, ignore_pixel=255):
        """
        :param input_logits: NxCxDxHxW
        :param gt_semantic_masks: NxCxDxHxW
        :return: final loss
        """
        N, C, D, H, W = input_logits.shape
        th = 1e-8  # 1e-10
        eps = 1e-10
        ignore_mask = (gts == ignore_pixel).detach()
        gt_semantic_masks = gts.detach()
        if torch.sum(ignore_mask):
            input_logits = torch.where(ignore_mask.view(N, 1, D, H, W).expand(N, self.num_classes, D, H, W),
                                       torch.zeros(N,C,D,H,W).cuda(),
                                       input_logits)
            gt_semantic_masks = torch.where(ignore_mask, torch.zeros(N,D,H,W).long().cuda(), gt_semantic_masks)
        gt_semantic_masks = _one_hot_embedding(gt_semantic_masks, self.num_classes).detach()
        print("gt_seantic_masks shape: ", gt_semantic_masks.size())

        g = _gumbel_softmax_sample(input_logits.view(N, C, -1), tau=0.5).reshape((N, C, D, H, W))
        g = compute_grad_mag(g, cuda=self._cuda)
 
        g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)

        g = g.view(N, -1)
        g_hat = g_hat.view(N, -1)
        loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return total_loss

def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    assert h > 1 and w > 1
    x, y = gradient_central_diff(input, cuda)
    return x, y

def convTri(input, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:
    """
    if (r <= 1):
        raise ValueError()
    n, c, h, w = input.shape
    return input
    f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
    kernel = torch.Tensor([f]) / (r + 1) ** 2
    if type(cuda) is int:
        if cuda != -1:
            kernel = kernel.cuda(device=cuda)
    else:
        if cuda is True:
            kernel = kernel.cuda()

    # padding w
    input_ = F.pad(input, (1, 1, 0, 0), mode='replicate')
    input_ = F.pad(input_, (r, r, 0, 0), mode='reflect')
    input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    input_ = torch.cat(input_, 3)
    t = input_

    # padding h
    input_ = F.pad(input_, (0, 0, 1, 1), mode='replicate')
    input_ = F.pad(input_, (0, 0, r, r), mode='reflect')
    input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    input_ = torch.cat(input_, 2)

    output = F.conv2d(input_,
                      kernel.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    output = F.conv2d(output,
                      kernel.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    return output


def compute_grad_mag(E, cuda=False):
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    mag = torch.sqrt(torch.mul(Ox,Ox) + torch.mul(Oy,Oy) + 1e-6)
    mag = mag / mag.max();

    return mag

def gradient_central_diff(input, cuda):
    return input, input
    kernel = [[1, 0, -1]]
    kernel_t = 0.5 * torch.Tensor(kernel) * -1.  # pytorch implements correlation instead of conv
    if type(cuda) is int:
        if cuda != -1:
            kernel_t = kernel_t.cuda(device=cuda)
    else:
        if cuda is True:
            kernel_t = kernel_t.cuda()
    n, c, h, w = input.shape

    x = conv2d_same(input, kernel_t.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    y = conv2d_same(input, kernel_t.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    return x, y

