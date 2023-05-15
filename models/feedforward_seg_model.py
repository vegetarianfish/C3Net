import torch
import torch.optim as optim
import torch.nn as nn

from collections import OrderedDict
import utils.util as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor
import numpy as np

class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain
        # self.gpu_ids = opts.gpu_ids
        if torch.cuda.device_count() > 0:
            self.gpu_ids = tuple(range(torch.cuda.device_count()))
            print("Use", self.gpu_ids, "GPUs.")
            # self.gpu_ids = None
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim
        self.ignore_index = opts.ignore_index
        self.void_classes = opts.void_classes
        self.model_type = opts.model_type

        # self.edge = True if self.model_type  else False
        # self.edge_type = opts.edge_type if self.model_type == "unet_ct_v_edge" else None
        if self.model_type == "unet_ct_v_edge" or self.model_type == "LMYNet_no_MuSeCWA" or self.model_type == "LMYNet_no_CWASPP":
            self.edge, self.edge_input, self.edge_type = True, opts.edge_input, opts.edge_type
            self.to_steal, self.loss_weights = 0.0, [1, 1]
        else:
            self.edge, self.edge_input, self.edge_type = False, False, None

        if 'boundary' in opts.criterion:
            self.boundary = True
            self.to_steal, self.loss_weights = 0.01, [1, 0.01]
            if self.edge: raise NotImplementedError
        # dice idc [0, 1]; boundary idc [1]
        # update every epoch
        else:
            self.boundary = False

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc, in_channels=opts.input_nc, aspp_channel=opts.aspp_channel,
                               nonlocal_mode=opts.nonlocal_mode, tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample, res_connect=opts.res_connect,
                               dropout=opts.dropout, gpu_ids=self.gpu_ids, edge_input=self.edge_input, edge_type=self.edge_type)
        # , ignore_index=opts.ignore_index, void_classes=opts.void_classes)
        # if opts.model_type != 'unet_ct_v_edge':
        self.net.cuda()
        # self.net.to(self.device)
            # if torch.cuda.device_count() > 1:
            #     self.net = torch.nn.DataParallel(self.net)

        # training objective
        self.criterion = get_criterion(opts)
        # initialize optimizers
        self.schedulers = []
        #  self.optimizer_vnet = get_optimizer(opts, self.net.gated_vnet.parameters())
        #  self.optimizer_gru = get_optimizer(opts, self.net.parameters(), lr=1e-4)
        self.init_lr = opts.lr_rate

        self.optimizers = []
        self.optimizer_vnet = get_optimizer(opts, [param for name, param in self.net.named_parameters()]) # if 'GRU' not in name])
        self.optimizers.append(self.optimizer_vnet)
        # if 'edge' in opts.model_type:
        #     self.optimizer_gru = get_optimizer(opts, [param for name, param in self.net.named_parameters() if 'GRU' in name])
        #     self.optimizers.append(self.optimizer_gru)

        # for name, param in self.net.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)

        if opts.continue_train:
            # self.path_pre_trained_model = opts.path_pre_trained_model
            # if self.path_pre_trained_model:
            #     self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
            #     self.which_epoch = int(0)
            # else:
            self.which_epoch = opts.which_epoch
            self.load_network(self.net, 'S', self.which_epoch)
            self.load_optimizer(self.optimizer_vnet, '0', self.which_epoch)
            # if 'edge' in opts.model_type:
            #     self.load_optimizer(self.optimizer_gru, '1', self.which_epoch)
        else:
            self.which_epoch = int(0)

        # print the network details
        if kwargs.get('verbose', True):
            print('Network is initialized')
            print_network(self.net)

        self.vis = False

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        self.input, self.target, self.size = inputs
        if self.boundary:
            self.target, self.dist = self.target
        if self.use_cuda:
            if not self.edge:
                self.input = self.input.to(self.device)
                self.target = self.target.to(self.device)
                if self.boundary:
                    self.target = (self.target, self.dist.to(self.device))
            else:
                if self.edge_input: input, edge_image = self.input
                target, edge_target = self.target
                if self.edge_input:
                    self.input = (input.to(self.device), edge_image.to(self.device))
                else:
                    self.input = self.input.to(self.device)
                self.target = (target.to(self.device), edge_target.to(self.device))

        # self.input = np.random.randn(16, 1, 144, 144, 144).astype(np.float32)

        # if not self.edge:
        #     self.input = self.input.cuda() if self.use_cuda else self.input
        #     self.target = self.target.cuda() if self.use_cuda else self.target
        # else:
        #     input, edge_image = self.input
        #     target, edge_target = self.target
        #     input = input.cuda() if self.use_cuda else input
        #     target = target.cuda() if self.use_cuda else target
        #     edge_image = edge_image.cuda() if self.use_cuda else edge_image
        #     edge_target = edge_target.cuda() if self.use_cuda else edge_target
        #     self.input = (input, edge_image)
        #     self.target = (target, edge_target)

        # for idx, _input in enumerate(inputs):
        #     # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
        #     bs = _input.size()
        #     if (self.tensor_dim == '2D') and (len(bs) > 4):
        #         _input = _input.permute(0,4,1,2,3).contiguous().view(bs[0]*bs[4], bs[1], bs[2], bs[3])
        #
        #     # Define that it's a cuda array
        #     if idx == 0:
        #         self.input = _input.cuda() if self.use_cuda else _input
        #     elif idx == 1:
        #         self.target = _input.cuda() if self.use_cuda else _input
        #         assert self.input.size() == self.target.size()

    def forward(self, split):
        if split == 'train':
            self.prediction = self.net(self.input)
        elif split == 'test':
            self.prediction = self.net(self.input)
            # Apply a softmax and return a segmentation map
            if self.edge:
                self.logits = self.net.apply_argmax_softmax(self.prediction[0])
            else:
                self.logits = self.net.apply_argmax_softmax(self.prediction)
            self.pred_seg = self.logits.detach().max(1)[1].unsqueeze(1)
            if self.vis: self.edge_att_list = self.prediction[2]
            
    def backward(self):
        self.loss_S = self.criterion(self.prediction, self.target)
        if not self.edge and not self.boundary:
            self.loss_S.backward()
        else:
            self.region_loss = self.loss_S['region_loss'].item()
            if self.edge:
                self.loss_S['region_loss'] *= self.loss_weights[0]
                self.loss_S['edge_loss'] *= self.loss_weights[1]
                # self.loss_S['align_loss'] *= self.loss_weights[2]
                self.edge_loss = self.loss_S['edge_loss'].item()
            if self.boundary:
                self.loss_S['region_loss'] *= self.loss_weights[0]
                self.loss_S['boundary_loss'] *= self.loss_weights[1]
                self.boundary_loss = self.loss_S['boundary_loss'].item()
            self.main_loss = 0
            for key in self.loss_S.keys():
                self.main_loss += self.loss_S[key]
            # self.main_loss = self.dice_loss + self.edge_loss
            self.main_loss.backward()

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def steal_weight(self):
        self.loss_weights = [max(0.1, self.loss_weights[0] - self.to_steal),
                             # max(0.1, self.loss_weights[1] - self.to_steal),
                             min(1, self.loss_weights[1] + self.to_steal)]

        print(f"Loss weights went to {self.loss_weights}")

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        # self.seg_scores, self.dice_score, self.hd95 = segmentation_stats(self.prediction, self.target, ignore_index=self.ignore_index)
        # seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        # print('len(self.prediction):', len(self.prediction))
        # print(self.prediction[0].shape)
        # print(self.prediction[1][0].shape)
        # print('len(self.target):', len(self.target))
        # print(self.target[0].shape)
        # print(self.target[1][0].shape)

        # print('self.ignore_index:', self.ignore_index)
        # print('GHJ CHECK!')
        # print(self.prediction.shape, self.target.shape)
        self.dice_score, self.hd95 = segmentation_stats(self.prediction, self.target, ignore_index=self.ignore_index)
        # self.dice_score, self.hd95 = np.zeros((15), dtype=np.float32), np.zeros((15), dtype=np.float32)
        seg_stats = []
        if self.void_classes is not None and 0 in self.void_classes:
            start_avg = 0
            # if ignored, dice_score should not have its indice
        else:
            seg_stats.append(('Dice_0', self.dice_score[0]))
            seg_stats.append(('HD95_0', self.hd95[0]))
            start_avg = 1
        # print("start_avg, self.dice_score.size:", start_avg, self.dice_score.size)
        for class_id in range(start_avg, self.dice_score.size):
            seg_stats.append(('Dice_{}'.format(class_id), self.dice_score[class_id]))
            seg_stats.append(('HD95_{}'.format(class_id), self.hd95[class_id]))
        dice_avg = np.nanmean(self.dice_score[start_avg:])
        hd95_avg = np.nanmean(self.hd95[start_avg:])
        seg_stats.insert(0, ('Dice_avg', dice_avg))
        seg_stats.insert(1, ('HD95_avg', hd95_avg))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        if self.edge:
            return OrderedDict([('region_Loss', self.region_loss),
                                ('edge_Loss', self.edge_loss)])
        elif self.boundary:
            return OrderedDict([('region_Loss', self.region_loss),
                                ('boundary_Loss', self.boundary_loss)])
        else:
            return OrderedDict([('Seg_Loss', self.loss_S.item())])
        # not good, should seperate different losses

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input, 'img')
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(self.input)

    # returns the fp/bp times of the model
    def get_fp_bp_time (self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        # inp_array = torch.zeros(*size).cuda()
        # out_array = torch.zeros(*size).cuda()
        inp_array = self.input.new_zeros(*size).cuda()
        out_array = self.target.new_zeros(*size).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids, self.optimizers)
        # if self.model_type == "unet_ct_v_edge":
        #     if len(self.gpu_ids) in [1, 2]:
        #         self.net.gated_vnet.cuda(0)
        #         if self.net.convGRU: self.net.convGRU.cuda(0)
        #         if self.net.aspp_channel: self.net.aspp.cuda(1)
        #         self.net.final_seg.cuda(1)
        #     # elif len(self.gpu_ids) == 2:
        #     #     self.net.gated_vnet.cuda(0)
        #     #     if self.net.aspp_channel: self.net.aspp.cuda(1)
        #     #     self.net.final_seg.cuda(0)
        #     elif len(self.gpu_ids) == 4:
        #         self.net.gated_vnet = self.net.gated_vnet.to('cuda:0')
        #         if self.net.aspp_channel: self.net.aspp = self.net.aspp.to('cuda:2')
        #         self.net.final_seg = self.net.edge_thread.to('cuda:2')
        # else:
        if len(self.gpu_ids) and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

