import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import numpy as np
# from scipy.ndimage import distance_transform_edt as distance
import edt

def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=None):
        super(SoftDiceLoss, self).__init__()
        # self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        batch_size, _, d, h, w = target.shape
        input = F.softmax(input, dim=1)

        if self.ignore_index is not None:
            score = 0

            for ipt, tgt in zip(input, target):
                # c, d, h, w
                ipt = ipt.contiguous().view(self.n_classes, -1) # batch_size?
                tgt = tgt.view(-1)
                mask = tgt != self.ignore_index
                tgt = tgt[mask]
                if mask.sum() == 0:
                    score += torch.mean(input*0)
                    continue
                for c in range(self.n_classes):
                    # oneipt: dxhxw
                    oneipt = ipt[c, :][mask]
                    onetgt = (tgt==c).float()
                    flatloss = self.__LossFlat__(oneipt, onetgt, dim=())
                    score += flatloss
        else:
            input = input.view(batch_size, self.n_classes, -1)
            # target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
            target = class2one_hot(target, self.n_classes).float().view(batch_size, self.n_classes, -1)
            score = self.__LossFlat__(input, target, 2)

        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

    def __LossFlat__(self, input, target, dim=None):
        # per sample per class
        smooth = 0.01
        inter = torch.sum(input * target, dim) + smooth
        union = torch.sum(input, dim) + torch.sum(target, dim) + smooth

        score = torch.sum(2.0 * inter / union)
        # skip the batch and class axis for calculating Dice score!
        return score


class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class DiceLoss():
    def __init__(self, n_classes):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = list(range(n_classes))
        self.n_classes = n_classes

    def __call__(self, input, target):
        input = F.softmax(input, dim=1)
        target = class2one_hot(target, self.n_classes)

        assert simplex(input) and simplex(target)

        pc = input[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection = einsum("bcdwh,bcdwh->bc", pc, tc)
        union = (einsum("bcdwh->bc", pc) + einsum("bcdwh->bc", tc))

        smooth = 0.01
        divided = 1 - (2 * intersection + smooth) / (union + smooth)

        loss = divided.mean()

        return loss

class GeneralizedDice():
    def __init__(self, n_classes, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.n_classes = n_classes
        self.idc = list(range(n_classes))

    def __call__(self, input, target):

        input = F.softmax(input, dim=1)
        target = class2one_hot(target, self.n_classes)

        pc = input[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w = 1 / ((einsum("bcdwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection = w * einsum("bcdwh,bcdwh->bc", pc, tc)
        union = w * (einsum("bcdwh->bc", pc) + einsum("bcdwh->bc", tc))

        smooth = 0.01
        # divided = 1 - 2 * (intersection + smooth) / (union + smooth)
        divided = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean() / self.n_classes

        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, n_classes,  Focal=True, alpha=0.7, gamma = 0.75):
        super(FocalTverskyLoss, self).__init__()
        # self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        # self.ignore_index = ignore_index
        self.focal = Focal
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        batch_size, _, d, h, w = target.shape
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        # target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = class2one_hot(target, self.n_classes).float().view(batch_size, self.n_classes, -1)
        if self.focal:
            return self.focal_tversky(target, input)
        else:
            return self.tversky_loss(target, input)

    def tversky(self, y_true, y_pred):
        smooth = 1.
        true_pos = torch.sum(y_true * y_pred, 2)
        false_neg = torch.sum(y_true * (1 - y_pred), 2)
        false_pos = torch.sum((1 - y_true) * y_pred, 2)
        return torch.sum((true_pos + smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + smooth))

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky(y_true,y_pred)

    def focal_tversky(self, y_true,y_pred):
        pt_1 = self.tversky(y_true, y_pred)
        return torch.pow((1-pt_1), self.gamma)


class ACLoss():
    def __init__(self, n_classes, weight_length, weight_region_in, weight_region_out):
        self.n_classes = n_classes
        self.weight_length = weight_length
        self.weight_region_in = weight_region_in
        self.weight_region_out = weight_region_out

    def __call__(self, input, target):
    # y_pred = K.cast(y_pred, dtype = 'float64')

        input = F.softmax(input, dim=1)
        target = class2one_hot(target, self.n_classes).type(torch.float32)

        """
        lenth term
        """

        lenth, region_in, region_out = 0, 0, 0
        if self.weight_length > 0:
            x = input[:, :, 1:, :, :] - input[:, :, :-1, :, :]  # horizontal and vertical directions
            y = input[:, :, :, 1:, :] - input[:, :, :, :-1, :]
            z = input[:, :, :, :, 1:] - input[:, :, :, :, :-1]

            delta_x = x[:, :, 1:, :-2, :-2] ** 2
            delta_y = y[:, :, :-2, 1:, :-2] ** 2
            delta_z = z[:, :, :-2, :-2, 1:] ** 2
            # delta_u = K.abs(delta_x + delta_y + delta_z)

            epsilon = 0.00000001  # where is a parameter to avoid square root is zero in practice.
            lenth = self.weight_length * torch.sum(torch.sqrt(torch.abs(delta_x+delta_y+delta_z) + epsilon))  # equ.(11) in the paper

        """
        region term
        """
        C_1 = torch.ones(target.shape, dtype=torch.float32)
        C_2 = torch.zeros(target.shape, dtype=torch.float32)
        if torch.cuda.is_available():
            C_1 = C_1.cuda()
            C_2 = C_2.cuda()
        # C_1 = np.ones((256, 256))
        # C_2 = np.zeros((256, 256))

        if self.weight_region_in > 0:
            region_out = self.weight_region_in * torch.abs(torch.mean((C_1 - input) * ((target - C_2) ** 2)))  # equ.(12) in the paper
        if self.weight_region_out > 0:
            region_in = self.weight_region_out * torch.abs(torch.mean(input * ((target - C_1) ** 2)))  # equ.(12) in the paper

        return lenth + region_in + region_out

def sset(a, sub):
    return set(torch.unique(a.cpu()).numpy()).issubset(sub)

def simplex(t, axis=1):
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)
    # Returns True if two arrays are element-wise equal within a tolerance.
    # every position should has a class label, False when there is ignore_class

def one_hot(t, axis=1):
    return simplex(t, axis) and sset(t, [0, 1])

def class2one_hot(seg, C):
    if len(seg.shape) == 5:  # Only w, h, used by the dataloader
        seg = seg.squeeze(dim=1)
    # assert sset(seg, list(range(C)))
    # print(seg.shape) # [1, 1, 144, ...]

    b, d, w, h = seg.shape

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, d, w, h)
    # assert simplex(res, 1)
    assert sset(res, [0, 1])

    return res

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim() #5
        output_size = X_in.size() + torch.Size([self.depth]) # torch.Size([bs, 1, 144, 144, 144, 15])
        # num_element = X_in.numel() # bs*2985984=bs*144*144*144
        X_in = X_in.detach().long().view(-1)
        out = self.ones.index_select(0, X_in).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()  # [bs, 15, 144, 144, 144]

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

def one_hot2dist(seg):
    if seg.size()[0] > 1:
        raise NotImplementedError
    # assert one_hot(seg, axis=1)
    C = seg.shape[1]
    seg_np = seg.cpu().numpy()
    # print("C in one_hot2dist:", C) # 9

    res = np.zeros_like(seg_np)
    for c in range(C):
        posmask = seg_np[:, c].astype(bool).squeeze()

        if posmask.any():
            negmask = ~posmask
            # res[:, c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # # surface loss for seg: inner: negative (encourage), outer: positive, boundary: 0
            posweight = edt.edt(posmask)
            posweight[posweight == 1] = -1.0
            res[:, c] = edt.edt(negmask) * negmask + posweight * posmask
            # my loss for edge: inner: positive, outer: positive (discourage), boundary: -1 (encourage)
    return torch.from_numpy(res).float().cuda()

def manual_s(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    manual_s(1)
    depth=3
    batch_size=2
    # encoder = One_Hot(depth=depth).forward
    y = torch.LongTensor(batch_size, 1, 1, 3, 3).cuda().random_() % depth  # 4 classes,1x3x3 img
    y[:, :, :, 0, :] = 255
    # y_onehot = encoder(y)
    x = torch.randn(batch_size, depth, 1, 3, 3).float().cuda()
    dicemetric = SoftDiceLoss(n_classes=depth, ignore_index=255)
    score = dicemetric(x,y)
    print('x:', x)
    print('y:', y)
    print('score:', score)
    y_masked = y[:, :, :, 1:3, :].contiguous()
    x_masked = x[:, :, :, 1:3, :].contiguous()
    dicemetric_masked = SoftDiceLoss(n_classes=depth)
    score_masked = dicemetric_masked(x_masked,y_masked)
    print('x_m:', x_masked)
    print('y_m:', y_masked)
    print('score_m:', score_masked)
