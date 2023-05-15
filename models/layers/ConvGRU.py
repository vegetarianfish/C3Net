import os
import torch
from torch import nn
from models.networks.vnet_3D import passthrough
from numpy import hstack

# t is multi-scale feature
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, class_wise, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.depth, self.height, self.width = input_size
        self.padding = kernel_size// 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.class_wise = class_wise
        self.dtype = dtype

        convgroups = self.hidden_dim if class_wise else 1

        self.conv_gates = nn.Conv3d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    groups=convgroups,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv3d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              groups=convgroups,
                              padding=self.padding,
                              bias=self.bias)

        if class_wise:
            assert input_dim == hidden_dim
            self.slice_layer = SliceLayer()
            self.concat_layer = ConcatLayer()

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, self.depth, self.height, self.width).type(self.dtype)
        # return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        if self.class_wise: #@GHJ: True
            sliced_x_list = self.slice_layer(input_tensor)
            sliced_h_list = self.slice_layer(h_cur)
            current_list_output = [i for tup in zip(sliced_x_list, sliced_h_list) for i in tup]
            # list(hstack(list(zip(sliced_x_list, sliced_h_list))))

            combined = self.concat_layer(current_list_output, dim=1)
        else:
            combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        if self.class_wise:
            fmap_size = input_tensor.size()[2:]
            # combined = torch.split(combined, 2, dim=1)
            combined = torch.split(combined_conv, 2, dim=1)
            # print(combined[0].size()) # [1, 2, ...]
            combined = [i.reshape(1, 2, 1, *fmap_size) for i in combined]
            gamma, beta = torch.split(torch.cat(combined, dim=2), 1, dim=1)
            gamma, beta = gamma.squeeze(1), beta.squeeze(1)
        else:
            gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        if self.class_wise:
            sliced_reset_h_list = self.slice_layer(reset_gate*h_cur)
            current_list_output = [i for tup in zip(sliced_x_list, sliced_reset_h_list) for i in tup]
            # list(hstack(list(zip(sliced_x_list, sliced_reset_h_list))))

            combined = self.concat_layer(current_list_output, dim=1)
        else:
            combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm) # GRUUNet: LeakyRelu (0.2x if >= 0)

        # h_next = update_gate * h_cur + (1 - update_gate) * cnm
        h_next = update_gate * cnm + (1 - update_gate) * h_cur
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, res_connect=False, class_wise=False,
                 dropout=0, dtype=torch.cuda.FloatTensor, batch_first=False, bias=True, return_all_layers=False):
        """

        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        # :param alexnet_path: str
        #     pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.res_connect = res_connect #@GHJ: False
        self.class_wise = class_wise #@GHJ: True
        self.dos = [passthrough for _ in range(num_layers-1)]
        if dropout:
            self.dos = [nn.Dropout3d(p=dropout) for _ in range(num_layers-1)]
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        print(self.kernel_size)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=input_size,
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         class_wise=class_wise,
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        # if not self.batch_first:
        #     # (t, b, c, h, w) -> (b, t, c, h, w)
        #     input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)
        if self.batch_first:
            raise NotImplementedError

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor[0].size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = len(input_tensor) #@ghj:4
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            # for cur_time_input in cur_layer_input:
            for t in range(seq_len):
                cur_time_input = cur_layer_input[t]
                h = self.cell_list[layer_idx](input_tensor=cur_time_input, h_cur=h)
                if self.res_connect and t:
                    h += cur_time_input
                output_inner.append(h)
            if self.num_layers > 1 and layer_idx != (self.num_layers-1):
                output_inner = [self.dos[layer_idx](output_inner[t]) for t in range(seq_len)]

            # layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = output_inner

            layer_output_list.append(output_inner)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        # print(last_state_list[0].size()) # torch.Size([1, 9, 144, 144, 144])
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class SliceLayer(nn.Module):

    def __init__(self):
        super(SliceLayer, self).__init__()

    def forward(self, input_data):
        """
        slice into several single piece in a specific dimension. Here for dim=1
        """
        sliced_list = []
        for idx in range(input_data.size()[1]):
            sliced_list.append(input_data[:, idx, :, :].unsqueeze(1))

        return sliced_list

class ConcatLayer(nn.Module):

    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, input_data_list, dim):
        concat_feats = torch.cat((input_data_list), dim=dim)
        return concat_feats


if __name__ == '__main__':
    # set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor # computation in GPU
    else:
        dtype = torch.FloatTensor

    height = width = 6
    channels = 256
    hidden_dim = [32, 64]
    kernel_size = (3,3) # kernel size for two stacked hidden layer
    num_layers = 2 # number of stacked hidden layer
    model = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=dtype,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)

    batch_size = 1
    time_steps = 1
    input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
    layer_output_list, last_state_list = model(input_tensor)