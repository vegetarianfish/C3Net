import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D
from .vnet_3D import InputTransition, DownTransition, UpTransition, OutputTransition, ContBatchNorm3d, ConvDropoutReLU, passthrough
from models import resnet
from models import GatedSpatialConv as gsc

FLAG_NON_SQUEEZE_GSC_EDGE = True
FLAG_VNET_OUTPUT_SEG = True
# edge down
FLAG_EDGE_NO_BG = False
FLAG_SCALE_GROUP = False
FLAG_EDGE_THRESH = False
FLAG_GATING_DO = 0
FLAG_MuSeAtt = True
# GRU
FLAG_EDGE_DEEP_SUP = True
FLAG_EDGE_GRU = True and FLAG_EDGE_DEEP_SUP
FLAG_EDGE_GRU_RES = True and FLAG_EDGE_GRU
FLAG_EDGE_GRU_CW = True # class-wise
# ASPP
FLAG_SHARED_CONCAT = True # if use 9 ch vnet_seg and 9 ch edge_out (no edge_input) as ASPP input
FLAG_SHARED_FUSE = True # if shared concat and all has background (or not) and aspp_channel==9 (or 8?)
# Final
FLAG_SIMPLE_FINAL_SEG = True
FLAG_DCRF = False


if FLAG_DCRF:
    from models.layers.dcrf3d import CrfRnn3d
    # from models.layers.crf import DenseCRF3D
if FLAG_EDGE_GRU:
    from models.layers.ConvGRU import ConvGRU

# 10060 / 10989 : conv0 and conv2 as gating signals
class unet_CT_V_edge_3D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, aspp_channel=16,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True,
                 res_connect=False, dropout=0, gpu_ids=(0,), edge_input=True, edge_type='binary'):
        # , ignore_index=255, void_classes=None):
        super(unet_CT_V_edge_3D, self).__init__()

        if FLAG_SHARED_CONCAT: assert FLAG_VNET_OUTPUT_SEG and (not edge_input) and edge_type.startswith('semantic') and FLAG_NON_SQUEEZE_GSC_EDGE
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.gpu_ids = gpu_ids
        self.aspp_channel = aspp_channel - int(FLAG_EDGE_NO_BG)

        # self.edge_input = edge_input
        self.edge_type = edge_type
        if self.edge_type.startswith('binary'):
            self.edge_channel = 1
        elif self.edge_type.startswith('semantic'):
            self.edge_channel = n_classes - int(FLAG_EDGE_NO_BG) # ignore the background class ?? nope
        else:
            raise NotImplementedError

        self.gated_vnet = Gated_VNet(feature_scale=feature_scale, n_classes=n_classes, is_deconv=is_deconv,
                                     in_channels=in_channels, nonlocal_mode=nonlocal_mode, attention_dsample=attention_dsample,
                                     is_batchnorm=is_batchnorm, res_connect=res_connect, dropout=dropout,
                                     edge_input=edge_input, edge_type=edge_type, edge_channel=self.edge_channel)

        if FLAG_EDGE_GRU:
            self.convGRU = ConvGRU(input_size=(144, 144, 144),
                                   input_dim=self.edge_channel, hidden_dim=[self.edge_channel]*1, # hidden_dim is output dim
                                   kernel_size=[3], num_layers=1, res_connect=FLAG_EDGE_GRU_RES, class_wise=FLAG_EDGE_GRU_CW, dropout=0,
                                   dtype=torch.cuda.FloatTensor if len(gpu_ids) >= 1 else torch.FloatTensor,
                                   batch_first=False, bias=True, return_all_layers=False)

        aspp_vnet_ndim = n_classes - int(FLAG_EDGE_NO_BG) if FLAG_VNET_OUTPUT_SEG else filters[1]
        aspp_edge_ndim = self.edge_channel + int(edge_input) if FLAG_NON_SQUEEZE_GSC_EDGE else 1

        if self.aspp_channel:
            self.aspp = _AtrousSpatialPyramidPoolingModule(aspp_vnet_ndim, aspp_edge_ndim, reduction_dim=self.aspp_channel,
                                                           output_stride=16, rates=[4, 8, 12])
            final_seg_channel = self.aspp_channel*(5+int(not FLAG_SHARED_CONCAT))
        else:
            assert not edge_input
            final_seg_channel = aspp_vnet_ndim # + aspp_edge_ndim

        if FLAG_SIMPLE_FINAL_SEG:
            # group = aspp_channel if FLAG_SHARED_FUSE else 1
            finalgroup = self.aspp_channel if FLAG_SHARED_FUSE and FLAG_SHARED_CONCAT else 1
            self.final_seg = nn.Sequential(
                nn.Conv3d(final_seg_channel, self.aspp_channel, kernel_size=1,
                          groups=finalgroup, bias=False))
                # nn.BatchNorm3d(256),
                # nn.ReLU(inplace=True),
                # nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm3d(256),
                # nn.ReLU(inplace=True),
                # nn.Conv3d(256, n_classes, kernel_size=1, bias=False))
        else:
            self.final_seg = FinalSeg(n_classes, aspp_channel)

        if FLAG_DCRF:
            self.crfrnn = CrfRnn3d(num_labels=n_classes, num_iterations=2)

        # if len(gpu_ids) == 1:
        #     self.gated_vnet.cuda(0)
        #     if FLAG_EDGE_GRU: self.convGRU.cuda(0)
        #     if self.aspp_channel: self.aspp.cuda(0)
        #     if self.final_seg: self.final_seg.cuda(0)
        #     if FLAG_DCRF: self.crfrnn.cuda(0)
        # elif len(gpu_ids) == 2:
        #     # self.gated_vnet = nn.DataParallel(self.gated_vnet, device_ids=[0, 1]).to('cuda:0')
        #     # # if self.final_seg: self.final_seg = nn.DataParallel(self.final_seg, device_ids=[0, 1]).to('cuda:0')
        #     # if FLAG_EDGE_GRU: self.convGRU = nn.DataParallel(self.convGRU, device_ids=[0, 1]).to('cuda:0')
        #     self.gated_vnet.cuda(0)
        #     if FLAG_EDGE_GRU: self.convGRU.cuda(0)
        #     if self.aspp_channel: self.aspp.cuda(1)
        #     if self.final_seg: self.final_seg.cuda(1)
        #     if FLAG_DCRF: self.crfrnn.cuda(0)


        # if len(gpu_ids) > 1:
        #     print("Model is split in 2 gpus.")
        #     self.aspp.cuda(1)
        #     self.bot_aspp.cuda(1)
        #     self.bot_fine.cuda(1)
        #     self.final_seg.cuda(1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

        self.vis = False

    def forward(self, inputs):

        if self.vis:
            vnet_seg, _, conv1, edges, sdown, gating = self.gated_vnet(inputs)
            edge_att = [i.detach().clone() for i in edges]
            sdown = [i.detach().clone() for i in sdown]
            gating = [i.detach().clone() for i in gating]
            sdown.insert(0, vnet_seg.detach().clone())
            sdown += gating
            sdown += edge_att
        else:
            vnet_seg, _, conv1, edges = self.gated_vnet(inputs)
        if FLAG_EDGE_GRU:
            _, edges = self.convGRU(edges)
            if self.vis:
                sdown.append(edges[0].detach().clone())
        if self.aspp_channel:
            # vnet_seg, conv1 = vnet_seg.cuda(1), conv1.cuda(1)
            if FLAG_EDGE_NO_BG: vnet_bg, vnet_seg = torch.split(vnet_seg, [1, self.edge_channel], dim=1)
            assert len(edges) == 1
            # edges = [i.cuda(1) for i in edges]
            if FLAG_EDGE_THRESH:
                edges_in = self.apply_argmax_softmax(edges[0])
                super_idx = edges_in > 0.9
                super_idx[:, 0, :, :, :] = True
                edges_in = torch.where(super_idx, edges_in, torch.zeros_like(edges_in))
                x = self.aspp(vnet_seg, edges_in)
            else:
                x = self.aspp(vnet_seg, edges[0])
            if FLAG_SIMPLE_FINAL_SEG: #@GHJ: True
                seg_out = self.final_seg(x)
            else:
                # x_size = vnet_seg.size()
                seg_out = self.final_seg(x, conv1, edges[-1])
            if FLAG_EDGE_NO_BG: seg_out = torch.cat([vnet_bg, seg_out], dim=1)
            # seg_out = seg_out.cuda(0)
            # edges = [i.cuda(0) for i in edges]
            # print(len(edges))  # 1
        else: seg_out = vnet_seg
        if FLAG_DCRF:   seg_out = self.crfrnn(inputs, seg_out)

        if self.vis: return seg_out, edges, sdown
        return seg_out, edges


        # print(up1.shape)
        # device = vnet_seg.device.index
        # if len(self.gpu_ids) > 1:
        #     move = int(len(self.gpu_ids) / 2)
        #     vnet_seg = vnet_seg.cuda(device + move)
        #     gsc_edge = gsc_edge.cuda(device + move)
        #     conv1 = conv1.cuda(device + move)
        # with open('up1.p', 'wb') as f:
        #     pickle.dump(up1.detach().cpu().numpy(), f)
        # up1 = up1[:, :16, :, :, :]
        # conv1 = F.interpolate(conv1, x_size[2:], mode='trilinear', align_corners=True)
        # if len(self.gpu_ids) > 1:
        #     x, conv1 = x.cuda(device), conv1.cuda(device)

        # seg_out = self.edge_thread(x)
        # seg_out = F.interpolate(seg_out, x_size[2:], mode='trilinear')

        # return seg_out, edges

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

class Gated_VNet(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True,
                 res_connect=False, dropout=0, edge_input=True, edge_type='binary', edge_channel=20):
        # , ignore_index=255, void_classes=None):
        super(Gated_VNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.dropout = dropout
        self.res_connect = res_connect
        self.edge_type = edge_type
        self.edge_input = edge_input
        self.edge_channel = edge_channel
        self.val_cnt = 0
        self.vis = False

        filters = [64, 128, 256, 512, 1024] # [16, 32, 64, 128, 256]
        filters = [int(x / self.feature_scale) for x in filters]
        # filters = [64, 128, 256, 512, 1024] // 4
        downlayers = [1, 2, 3, 4, 3]
        uplayers = [3, 3, 2, 2]

        # downsampling
        self.conv0 = InputTransition(self.in_channels, filters[0], downlayers[0]-1, elu=False)

        self.conv1 = DownTransition(filters[0], filters[1], downlayers[1]-1, elu=False, dropout=0)

        self.conv2 = DownTransition(filters[1], filters[2], downlayers[2]-1, elu=False, dropout=0)

        self.conv3 = DownTransition(filters[2], filters[3], downlayers[3]-1, elu=False, dropout=self.dropout)

        self.conv4 = DownTransition(filters[3], filters[4], downlayers[4]-1, elu=False, dropout=self.dropout)

        # upsampling
        self.up_concat4 = UpTransition(filters[4], filters[4], nConvs=uplayers[0]-1, elu=False, dropout=self.dropout, SE=True)
        self.up_concat3 = UpTransition(filters[4], filters[3], nConvs=uplayers[1]-1, elu=False, dropout=self.dropout, SE=True)
        self.up_concat2 = UpTransition(filters[3], filters[2], nConvs=uplayers[2]-1, elu=False, dropout=0, SE=True)
        self.up_concat1 = UpTransition(filters[2], filters[1], nConvs=uplayers[3]-1, elu=False, dropout=0, SE=True)
        if FLAG_VNET_OUTPUT_SEG:
            self.output = OutputTransition(filters[1], n_classes, elu=False)

        # gating signals
        self.dsn_down1 = ConvDropoutReLU(filters[1], edge_channel, 1, dropout=FLAG_GATING_DO)
        self.dsn_down2 = ConvDropoutReLU(filters[2], edge_channel, 1, dropout=FLAG_GATING_DO)
        self.dsn_down3 = ConvDropoutReLU(filters[3], edge_channel, 1, dropout=FLAG_GATING_DO)
        self.dsn_down4 = ConvDropoutReLU(filters[4], edge_channel, 1, dropout=FLAG_GATING_DO)
        if FLAG_MuSeAtt:
            self.dsn_up4 = ConvDropoutReLU(filters[4], edge_channel, 1, dropout=FLAG_GATING_DO)
            self.dsn_up3 = ConvDropoutReLU(filters[4], edge_channel, 1, dropout=FLAG_GATING_DO)
            self.dsn_up2 = ConvDropoutReLU(filters[3], edge_channel, 1, dropout=FLAG_GATING_DO)
            self.dsn_up1 = ConvDropoutReLU(filters[2], edge_channel, 1, dropout=FLAG_GATING_DO)

        # self.dsn_down4 = nn.Conv3d(filters[4], 1, 1)
        # self.dsn_up3 = nn.Conv3d(filters[4], 1, 1) # dsn_up3(up4)
        # self.dsn_up4 = nn.Conv3d(filters[2], 1, 1) # dsn_up4(up2)
        # self.dsn_up0 = nn.Conv3d(n_classes, 1, 1)

        # self.res1 = resnet.BasicBlock(edge_channel, edge_channel, stride=1, downsample=None)
        # self.d1 = nn.Sequential(nn.Conv3d(edge_channel, edge_channel, kernel_size=1, bias=False),
        #                   nn.ReLU(inplace=True))
        # self.res2 = resnet.BasicBlock(edge_channel, edge_channel, stride=1, downsample=None)
        # self.d2 = nn.Sequential(nn.Conv3d(edge_channel, edge_channel, kernel_size=1, bias=False),
        #                         nn.ReLU(inplace=True))
        # self.res3 = resnet.BasicBlock(edge_channel, edge_channel, stride=1, downsample=None)
        # self.d3 = nn.Sequential(nn.Conv3d(edge_channel, edge_channel, kernel_size=1, bias=False),
        #                         nn.ReLU(inplace=True))

        # self.res1 = resnet.BasicBlock(2 * edge_channel, 2 * edge_channel, stride=1, downsample=None)
        # self.d1 = nn.Conv3d(2 * edge_channel, edge_channel, 1)
        # self.res2 = resnet.BasicBlock(2 * edge_channel, 2 * edge_channel, stride=1, downsample=None)
        # self.d2 = nn.Conv3d(2 * edge_channel, edge_channel, 1)
        # self.res3 = resnet.BasicBlock(2 * edge_channel, 2 * edge_channel, stride=1, downsample=None)
        # self.d3 = nn.Conv3d(2 * edge_channel, edge_channel, 1)

        # shape forward flows
        # start_filters = filters[1]
        # self.res1 = resnet.BasicBlock(start_filters, start_filters, stride=1, downsample=None)
        # self.d1 = nn.Conv3d(start_filters, start_filters//2, 1)
        # self.gate1 = gsc.GatedSpatialConv3d(start_filters//2, start_filters//2)
        # self.res2 = resnet.BasicBlock(start_filters//2, start_filters//2, stride=1, downsample=None)
        # self.d2 = nn.Conv3d(start_filters//2, start_filters//4, 1)
        # self.gate2 = gsc.GatedSpatialConv3d(start_filters//4, start_filters//4)
        # self.res3 = resnet.BasicBlock(start_filters//4, start_filters//4, stride=1, downsample=None)
        # self.d3 = nn.Conv3d(start_filters//4, start_filters//8, 1)
        # self.gate3 = gsc.GatedSpatialConv3d(start_filters//8, start_filters//8)

        # self.edge_fuse = nn.Conv3d(start_filters//8, edge_channel, 1, padding=0, bias=False)
        #
        if not FLAG_NON_SQUEEZE_GSC_EDGE:
            raise NotImplementedError

        self.sigmoid = nn.Sigmoid()

        # final conv (without any concat)
        # self.final = nn.Conv3d(n_classes*4, n_classes, 1)
            # Feature Extraction

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        if self.edge_input: inputs, edges = inputs
        x_size = inputs.size()

        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # print(conv0.shape, conv1.shape, conv2.shape, conv3.shape, conv4.shape)

        up4 = self.up_concat4(conv3, conv4)
        up3 = self.up_concat3(conv2, up4)
        up2 = self.up_concat2(conv1, up3)
        up1 = self.up_concat1(conv0, up2)

        # up0 = self.output(up1)

        s_down1 = self.dsn_down1(conv1)
        s_down2 = self.dsn_down2(conv2)
        s_down4 = self.dsn_down4(conv4)
        if FLAG_MuSeAtt:
            gating1 = torch.sigmoid(self.dsn_up1(up2))
            gating2 = torch.sigmoid(self.dsn_up2(up3))
            gating4 = torch.sigmoid(self.dsn_up4(conv4))
            s_down1_gated = F.interpolate(s_down1 * gating1, x_size[2:], mode='trilinear', align_corners=True)
            s_down2_gated = F.interpolate(s_down2 * gating2, x_size[2:], mode='trilinear', align_corners=True)
            s_down4_gated = F.interpolate(s_down4 * gating4, x_size[2:], mode='trilinear', align_corners=True)
        else:
            s_down1_gated = F.interpolate(s_down1, x_size[2:], mode='trilinear', align_corners=True)
            s_down2_gated = F.interpolate(s_down2, x_size[2:], mode='trilinear', align_corners=True)
            s_down4_gated = F.interpolate(s_down4, x_size[2:], mode='trilinear', align_corners=True)

        if FLAG_SCALE_GROUP:
            # s_up4 = F.interpolate(self.dsn_up4(up4), x_size[2:], mode='trilinear', align_corners=True)
            s_up3 = F.interpolate(self.dsn_up3(up3), x_size[2:], mode='trilinear', align_corners=True)
            s_up2 = F.interpolate(self.dsn_up2(up2), x_size[2:], mode='trilinear', align_corners=True)
            s_up1 = F.interpolate(self.dsn_up1(up1), x_size[2:], mode='trilinear', align_corners=True)

            GroupLarge = [0, 1, 2, 5, 6, 7] # scales [0, 1, 5, 6, 7]
            GroupSmall = [3, 4, 8] # scales [0, 4, 5, 6, 7]
            LargeEdge = [s_down2, s_down4, s_up3, s_up1]
            SmallEdge = [s_down1, s_down2, s_up2, s_up1]
            for i in range(3):
                LargeEdge[i][:, GroupSmall, :, :, :] = SmallEdge[i][:, GroupSmall, :, :, :]
            LargeEdge = [F.softmax(i, dim=1) for i in LargeEdge]
        # s_down1, s_down2, s_up3, s_up2, s_up1 = \
        #     [F.softmax(i, dim=1) for i in [s_down1, s_down2, s_up3, s_up2, s_up1]]
        # s_down3, s_down4 = [torch.where(i > 0.5, i, torch.zeros_like(i)) for i in [s_down3, s_down4]]

        # if FLAG_EDGE_DEEP_SUP: edges_out = [s_down1, s_down2, s_up3, s_up2, s_up1]
            edges_out = LargeEdge
        else:
            s_down3 = self.dsn_down3(conv3)
            if FLAG_MuSeAtt:
                gating3 = torch.sigmoid(self.dsn_up3(up4))
                s_down3_gated = F.interpolate(s_down3 * gating3, x_size[2:], mode='trilinear', align_corners=True)
            else:
                s_down3_gated = F.interpolate(s_down3, x_size[2:], mode='trilinear', align_corners=True)
            # s_down1, s_down2, s_down3, s_down4 = \
            #     [F.softmax(i, dim=1) for i in [s_down1, s_down2, s_down3, s_down4]]
            edges_out = [s_down1_gated, s_down2_gated, s_down3_gated, s_down4_gated]
        # if FLAG_VIS:
        #     for scan_channel in range(0, 144, int(144 / 10)):
        #     input_img = inputs[0, 0, :, scan_channel].cpu().numpy()
        #     for layer in edges_out:
        #         for ch in range(self.edge_channel):
        #             # nib.save(nib.Nifti1Image(layer[0, ch], np.eye(4)), os.path.join(self.save_dir, 'img_{}_{}_{}.nii.gz'.format(self.val_cnt, layer, ch)))
        #             col_num = 4
        #             row_num = 3
        #             fig, axarr = plt.subplots(row_num, col_num, figsize=(12, 12))  # , sharex=True, sharey=True)
        #     self.val_cnt += 1


        # edgemaps = torch.cat((s_down1, s_down2), dim=1)
        # # edgemaps = torch.add(s_down1, s_down2)
        # edgemaps = F.interpolate(self.d1(self.res1(edgemaps)), x_size[2:], mode='trilinear', align_corners=True)
        # if FLAG_EDGE_DEEP_SUP: edges_out.append(edgemaps.clone().detach().requires_grad_(True))
        #
        # edgemaps = torch.cat((edgemaps, s_down3), dim=1)
        # edgemaps = F.interpolate(self.d2(self.res2(edgemaps)), x_size[2:], mode='trilinear', align_corners=True)
        # if FLAG_EDGE_DEEP_SUP: edges_out.append(edgemaps.clone().detach().requires_grad_(True))
        #
        # edgemaps = torch.cat((edgemaps, s_down4), dim=1)
        # edgemaps = F.interpolate(self.d3(self.res3(edgemaps)), x_size[2:], mode='trilinear', align_corners=True)
        # if FLAG_EDGE_DEEP_SUP:
        #     edges_out.append(edgemaps)
        # else:
        #     edges_out = [edgemaps]
        #
        # gsc_edge = self.sigmoid(edgemaps)

        # print(torch.all(gsc_edge.eq(edgemaps))) # 0
        # print(torch.all(edges_out[0].eq(edgemaps))) # 0
        # print(torch.all(edges_out[-1].eq(edgemaps))) # 1

        if FLAG_VNET_OUTPUT_SEG:
            vnet_seg = self.output(up1)
            if self.vis:
                if not FLAG_MuSeAtt: raise NotImplementedError
                return vnet_seg, None, conv1, edges_out, [s_down1, s_down2, s_down3, s_down4], [gating1, gating2, gating3, gating4]
            else:
                return vnet_seg, None, conv1, edges_out
        return up1, None, conv1, edges_out

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, edge_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.use_shared_concat = True if (FLAG_SHARED_CONCAT and in_dim == edge_dim) else False
        self.convgroups = edge_dim if (self.use_shared_concat and FLAG_SHARED_FUSE) else 1
        if self.use_shared_concat: in_dim = in_dim*2
        if self.convgroups != 1: assert reduction_dim == self.convgroups

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv3d(in_dim, reduction_dim, kernel_size=1, groups=self.convgroups, bias=False),
                          ContBatchNorm3d(reduction_dim),
                          # nn.InstanceNorm3d(reduction_dim, affine=True),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv3d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, groups=self.convgroups, bias=False),
                ContBatchNorm3d(reduction_dim),
                # nn.InstanceNorm3d(reduction_dim, affine=True),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_avg_pooling = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_dim, reduction_dim, kernel_size=1, groups=self.convgroups, bias=False),
            # nn.InstanceNorm3d(reduction_dim, affine=True),
            # ContBatchNorm3d(reduction_dim),
            nn.ReLU(inplace=True))

        if not self.use_shared_concat:
            edge_input_channel = edge_dim # if FLAG_NON_SQUEEZE_GSC_EDGE else 1
            self.edge_conv = nn.Sequential(
                nn.Conv3d(edge_input_channel, reduction_dim, kernel_size=1, bias=False),
                ContBatchNorm3d(reduction_dim),
                # nn.InstanceNorm3d(reduction_dim, affine=True),
                nn.ReLU(inplace=True))
        else:
            self.slice_layer = SliceLayer()
            self.concat_layer = ConcatLayer()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, edge):
        x_size = x.size()
        if self.use_shared_concat: #@GHJ: True
            # input: VNet seg_out, output: semantic edge_out
            assert x_size[1] == edge.size()[1]
            sliced_x_list = self.slice_layer(x)
            sliced_edge_list = self.slice_layer(edge)

            # Add low-level feats to sliced_list
            sliced_list_inputs = [i for tup in zip(sliced_x_list, sliced_edge_list) for i in tup]
            # list(np.hstack(list(zip(sliced_x_list, sliced_edge_list))))
            # for i in range(len(sliced_edge_list)):
            #     sliced_list_inputs.append(sliced_x_list[i])
            #     sliced_list_inputs.append(sliced_edge_list[i])

            x = self.concat_layer(sliced_list_inputs, dim=1) # [BS, 18, ...]

        img_features = self.img_avg_pooling(x)
        # print('1:', img_features.shape)
        img_features = F.interpolate(img_features, x_size[2:], mode='trilinear', align_corners=True)
        # print('2:', img_features.shape)
        out = img_features

        if not self.use_shared_concat:
            edge_features = F.interpolate(edge, x_size[2:], mode='trilinear',align_corners=True)
            edge_features = self.edge_conv(edge_features)
            out = torch.cat((out, edge_features), 1)

        # print(out.size()) # torch.Size([1, 15, 144, 144, 144])

        if self.use_shared_concat:
            # group = 2 # out, GAP of seg + edge
            feat_fovs = []

        for f in self.features:
            y = f(x)
            if not self.use_shared_concat and self.convgroups == 1:
                out = torch.cat((out, y), 1)
            else:
                feat_fovs.append(y)
                # print('feat_fovs:', y.shape)

        if self.use_shared_concat:
            assert not torch.allclose(feat_fovs[0], feat_fovs[1])
            sliced_out_list = self.slice_layer(out)
            feat_fovs = [self.slice_layer(i) for i in feat_fovs]
            current_list_output = []
            for i in range(len(sliced_out_list)):
                current_list_output.extend([fov[i] for fov in feat_fovs])
                current_list_output.append(sliced_out_list[i])
                # print(len(current_list_output))
            out = self.concat_layer(current_list_output, dim=1)

        return out

class FinalSeg(nn.Module):
    def __init__(self, n_classes, aspp_channel, expansion=2):
        super(FinalSeg, self).__init__()
        # self.aspp= nn.Sequential(*list(self.aspp.children()))
        # self.aspp_last = list(self.aspp.children())[-1]
        self.n_classes = n_classes

        # self.bot_fine = nn.Conv3d(32, 1, kernel_size=1, bias=False)
        self.bot_fine = nn.Sequential(
            nn.ConvTranspose3d(32, 1, kernel_size=2, stride=2, bias=False),
            # nn.InstanceNorm3d(1, affine=True),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1, 1, kernel_size=1, bias=False)
        )
        # self.bot_aspp = nn.Conv3d(1280 + 256, 256, kernel_size=1, bias=False)
        # self.bot_aspp = nn.Conv3d(aspp_channel*(6 + int(not FLAG_SHARED_CONCAT)),
        #                           128, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv3d(aspp_channel*(6 + int(not FLAG_SHARED_CONCAT)), n_classes*expansion, kernel_size=3, padding=1, groups=n_classes, bias=False),
            nn.BatchNorm3d(n_classes*expansion),
            # nn.InstanceNorm3d(n_classes*expansion, affine=True),
            nn.ReLU(inplace=True),
            # nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm3d(256),
            # nn.ReLU(inplace=True),
            nn.Conv3d(n_classes*expansion, n_classes, kernel_size=1, groups=n_classes, bias=False))

        self.slice_layer = SliceLayer()
        self.concat_layer = ConcatLayer()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, x, conv1, edges):
        # x = self.aspp_m1(up1, acts)
        # x = x.view(x.size()[0], -1)
        # x = self.aspp_last(x)
        edges = torch.sigmoid(edges)
        dec0_fine = self.bot_fine(conv1)
        # print(dec0_fine.size())
        # dec0_fine= F.interpolate(dec0_fine, x.size()[2:], mode='trilinear', align_corners=True)

        sliced_x_list = self.slice_layer(x)
        current_list_output = []
        for i in range(self.n_classes):
            current_list_output.extend(sliced_x_list[i*5:(1+i)*5])
            current_list_output.append(dec0_fine*edges[:, i:i+1, :, :, :])
            # current_list_output.append(dec0_fine*edges[:, i:i+1, :, :, :])
        out = self.concat_layer(current_list_output, dim=1)

        # dec0_fine = self.bot_fine(conv1)
        # dec0_up = self.bot_aspp(x)
        # dec0_up = F.interpolate(dec0_up, conv1.size()[2:], mode='trilinear', align_corners=True)
        # dec0 = torch.cat((dec0_fine, dec0_up), dim=1)

        seg_out = self.final_seg(out)
        # seg_out = F.interpolate(dec1, size, mode='trilinear', align_corners=True)
        return seg_out

class SliceLayer(nn.Module):

    def __init__(self):
        super(SliceLayer, self).__init__()

    def forward(self, input_data):
        """
        slice into several single piece in a specific dimension. Here for dim=1
        """
        sliced_list = []
        for idx in range(input_data.size()[1]):
            sliced_list.append(input_data[:, idx, :, :, :].unsqueeze(1)) # the same as [:, idx]

        return sliced_list

class ConcatLayer(nn.Module):

    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, input_data_list, dim):
        concat_feats = torch.cat((input_data_list), dim=dim)
        return concat_feats

# class MultiAttentionBlock(nn.Module):
#     def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
#         super(MultiAttentionBlock, self).__init__()
#         self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
#                                                  inter_channels=inter_size, mode=nonlocal_mode,
#                                                  sub_sample_factor= sub_sample_factor)
#         self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
#                                                  inter_channels=inter_size, mode=nonlocal_mode,
#                                                  sub_sample_factor=sub_sample_factor)
#         self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
#                                            nn.BatchNorm3d(in_size),
#                                            nn.ReLU(inplace=True)
#                                            )
#
#         # initialise the blocks
#         for m in self.children():
#             if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
#             init_weights(m, init_type='kaiming')
#
#     def forward(self, input, gating_signal):
#         gate_1, attention_1 = self.gate_block_1(input, gating_signal)
#         gate_2, attention_2 = self.gate_block_2(input, gating_signal)
#
#         return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


