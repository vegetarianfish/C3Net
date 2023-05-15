'''
Misc Utility functions
'''

import os
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from utils.metrics import segmentation_scores, dice_score_list, hd95_list
from sklearn import metrics
from .layers.loss import *
from .layers.edge_loss import JointEdgeDiceLoss
from .layers.boundary_loss import JointDiceBoundaryLoss
from .layers.dice_ce_loss import DC_and_CE_loss

def get_optimizer(option, params, lr=0):
    opt_alg = 'sgd' if not hasattr(option, 'optim') else option.optim
    print("optim: ", opt_alg)
    if opt_alg == 'sgd':
        optimizer = optim.SGD(params,
                              lr=option.lr_rate,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=option.l2_reg_weight)
    #     nnunet used momentum=0.99

    elif opt_alg == 'adam':
        optimizer = optim.Adam(params,
                               lr=lr if lr else option.lr_rate,
                               betas=(0.9, 0.999),
                               weight_decay=option.weight_decay)

    return optimizer


def get_criterion(opts):
    if opts.criterion == 'cross_entropy':
        if opts.type == 'seg':
            criterion = cross_entropy_2D if opts.tensor_dim == '2D' else cross_entropy_3D
        elif 'classifier' in opts.type:
            criterion = CrossEntropyLoss()
    elif opts.criterion == 'dice_loss':
        criterion = SoftDiceLoss(opts.output_nc, opts.ignore_index)
        # criterion = GeneralizedDice(opts.output_nc)
        # criterion = DiceLoss(opts.output_nc)
    elif opts.criterion == 'dice_ce_loss':
        criterion = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, {})
    elif opts.criterion == 'focal_tversky_loss':
        criterion = FocalTverskyLoss(opts.output_nc, True)
    elif opts.criterion == 'tversky_loss':
        criterion = FocalTverskyLoss(opts.output_nc, False)
    elif opts.criterion == 'dice_boundary_loss':
        criterion = JointDiceBoundaryLoss(opts.output_nc, dice='softdiceloss')
        # raise NotImplementedError
    elif opts.criterion == 'dice_loss_pancreas_only':
        criterion = CustomSoftDiceLoss(opts.output_nc, class_ids=[0, 2])
    elif opts.criterion == 'edge_dice_loss': # @GHJ: This Way!
        criterion = JointEdgeDiceLoss(classes=opts.output_nc, region_loss='dice', ignore_index=opts.ignore_index,
                                      edge_weight=opts.edge_weight, dice_weight=opts.dice_weight,
                                      edge_type=opts.edge_type, edge_att_weight=opts.edge_att_weight,
                                      seg_grad_weight=opts.seg_grad_weight)
    elif opts.criterion == 'edge_tversky_loss':
        criterion = JointEdgeDiceLoss(classes=opts.output_nc, region_loss='tversky', ignore_index=opts.ignore_index,
                                      edge_weight=opts.edge_weight, dice_weight=opts.dice_weight,
                                      edge_type=opts.edge_type, edge_att_weight=opts.edge_att_weight,
                                      seg_grad_weight=opts.seg_grad_weight)
    elif opts.criterion == 'edge_focaltversky_loss':
        criterion = JointEdgeDiceLoss(classes=opts.output_nc, region_loss='focal_tversky', ignore_index=opts.ignore_index,
                                      edge_weight=opts.edge_weight, dice_weight=opts.dice_weight,
                                      edge_type=opts.edge_type, edge_att_weight=opts.edge_att_weight,
                                      seg_grad_weight=opts.seg_grad_weight)

    return criterion

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler(optimizers, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizers

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr*(1 - iter/max_iter)**power
            print('current learning rate = %.7f' % param_group['lr'])


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def segmentation_stats(pred_seg, target, ignore_index=None):
    if type(pred_seg) is tuple: pred_seg = pred_seg[0]
    if type(target) is tuple: target = target[0]
    n_classes = pred_seg.size(1)
    pred_lbls = pred_seg.detach().max(1)[1].cpu().numpy() # channel who has the maximum predicted logit
    gt = target.detach().cpu().numpy()
    # if target.size[1] == 1:
    #     gt = np.squeeze(target, axis=1)
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_)
        preds.append(pred_)

    iou = segmentation_scores(gts, preds, n_class=n_classes, ignore_index=ignore_index)
    # print('iou:', iou)
    dice = dice_score_list(gts, preds, n_class=n_classes, ignore_index=ignore_index)
    # print('dice:', dice)
    # print(len(gts), len(preds))
    gts[0] = np.squeeze(gts[0], axis=0)
    # print(gts[0].shape, preds[0].shape)
    hd95 = hd95_list(gts, preds, n_class=n_classes, ignore_index=ignore_index)
    # print(('hd95:', hd95))

    return dice, hd95


def classification_scores(gts, preds, labels):
    accuracy        = metrics.accuracy_score(gts,  preds)
    class_accuracies = []
    for lab in labels: # TODO Fix
        class_accuracies.append(metrics.accuracy_score(gts[gts == lab], preds[gts == lab]))
    class_accuracies = np.array(class_accuracies)

    f1_micro        = metrics.f1_score(gts,        preds, average='micro')
    precision_micro = metrics.precision_score(gts, preds, average='micro')
    recall_micro    = metrics.recall_score(gts,    preds, average='micro')
    f1_macro        = metrics.f1_score(gts,        preds, average='macro')
    precision_macro = metrics.precision_score(gts, preds, average='macro')
    recall_macro    = metrics.recall_score(gts,    preds, average='macro')

    # class wise score
    f1s        = metrics.f1_score(gts,        preds, average=None)
    precisions = metrics.precision_score(gts, preds, average=None)
    recalls    = metrics.recall_score(gts,    preds, average=None)

    confusion = metrics.confusion_matrix(gts,preds, labels=labels)

    #TODO confusion matrix, recall, precision
    return accuracy, f1_micro, precision_micro, recall_micro, f1_macro, precision_macro, recall_macro, confusion, class_accuracies, f1s, precisions, recalls


def classification_stats(pred_seg, target, labels):
    return classification_scores(target, pred_seg, labels)
