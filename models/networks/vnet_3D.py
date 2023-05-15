import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks_other import init_weights


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(nchan)
        # self.bn1 = nn.InstanceNorm3d(nchan, affine=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)

class InputTransition(nn.Module):
    def __init__(self, in_size, out_size, nConvs=0, elu=False, dropout=False):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=5, padding=2),
                                   # nn.InstanceNorm3d(out_size, affine=True),
                                   nn.BatchNorm3d(out_size),
                                   # ELUCons(elu, out_size),
                                   )
            # Conv3d(1, 16, kernel_size=5, padding=2)
        # self.relu = nn.ReLU()
        self.relu = ELUCons(elu, out_size)
        self.do1 = passthrough
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = passthrough
        if nConvs > 0:
            self.ops = _make_nConv(out_size, nConvs, elu)

        for m in self.children():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Norm') != -1 or classname.find('Linear') != -1:
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # do we want a PRELU here as well?
        out = self.conv1(inputs)
        out = self.do1(out)
        out = self.ops(out)
        # split input in to 16 channels
        # x16 = torch.cat([inputs]*self.out_size, 0)
        out = self.relu(torch.add(out, inputs[0,0,:].unsqueeze(0).unsqueeze(0))) #ghj
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=0.5):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        # self.bn1 = nn.InstanceNorm3d(outChans, affine=True)
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.do1 = passthrough
        if dropout:
            self.do1 = nn.Dropout3d(p=dropout)
        self.ops = _make_nConv(outChans, nConvs, elu)

        for m in self.children():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Norm') != -1 or classname.find('Linear') != -1:
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs=1, elu=False, dropout=0.5, SE=True):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        # self.bn1 = nn.InstanceNorm3d(outChans // 2, affine=True)
        self.do2 = nn.Dropout3d(p=dropout)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        self.do1 = passthrough
        if dropout:
            self.do1 = nn.Dropout3d(p=dropout)
        self.ops = _make_nConv(outChans, nConvs, elu)
        self.se = passthrough
        if SE:
            self.se = SELayer3D(outChans)   #self.se = SELayer3D(outChans // 2)

        for m in self.children():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Norm') != -1 or classname.find('Linear') != -1:
                init_weights(m, init_type='kaiming')

    def forward(self, skipx, x):
        out = self.do1(x)
        skipxdo = self.do2(skipx)   #skipxdo = self.do2(self.se(skipx))
        out = self.relu1(self.bn1(self.up_conv(out)))
        # print(x.shape, out.shape, skipxdo.shape)
        xcat = torch.cat((out, skipxdo), 1)
        out = self.se(self.ops(xcat))
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        # self.bn1 = nn.InstanceNorm3d(outChans, affine=True)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)
        # if nll:
        #     self.softmax = F.log_softmax
        # else:
        #     self.softmax = F.softmax

        for m in self.children():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Norm') != -1 or classname.find('Linear') != -1:
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # # treat channel 0 as the predicted output
        return out

class ConvDropoutReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout=0):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(ConvDropoutReLU, self).__init__()

        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size)

        # maybe dropout
        if dropout:
            self.do = nn.Dropout3d(p=dropout)

        # if norm == 'IN':
        #     self.norm = nn.InstanceNorm3d(output_channels, affine=True)
        # elif norm == 'BN':
        #     self.norm = nn.BatchNorm3d(output_channels)
        # else:
        #     raise NotImplementedError
            # self.norm = passthrough

        self.nonlin = nn.LeakyReLU(inplace=True)

        if dropout:
            self.all = nn.Sequential(self.conv, self.do, self.nonlin)
        else:
            self.all = nn.Sequential(self.conv, self.nonlin)


    def forward(self, x):
        return self.all(x)


class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out