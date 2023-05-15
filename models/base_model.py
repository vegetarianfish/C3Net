import os
import numpy
import torch
from utils.util import mkdir
from .networks_other import get_n_parameters
from collections import OrderedDict
from .utils import poly_lr_scheduler

class BaseModel():
    def __init__(self):
        self.input = None
        self.net = None
        self.isTrain = False
        self.use_cuda = True
        self.schedulers = []
        self.optimizers = []
        self.save_dir = None
        self.gpu_ids = []
        self.which_epoch = int(0)
        self.path_pre_trained_model = None

    def name(self):
        return 'BaseModel'

    def initialize(self, opt, **kwargs):
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.ImgTensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.LblTensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = opt.save_dir; mkdir(self.save_dir); print("Writing into ", self.save_dir)


    def set_input(self, input):
        self.input = input

    def set_scheduler(self, train_opt):
        pass

    def forward(self, split):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def get_input_size(self):
        return self.input.size() if input else None

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids, optimizers):
        print('Saving the model {0} at the end of epoch {1}'.format(network_label, epoch_label))
        save_filename_opt = []
        if isinstance(epoch_label, int):
            save_filename_net = '{0:03d}_net_{1}.pth'.format(epoch_label, network_label)
            for name, opt in optimizers:
                save_filename_opt.append('{0:03d}_opt_{1}.pth'.format(epoch_label, name))
        else:
            save_filename_net = '{0}_net_{1}.pth'.format(epoch_label, network_label)
            for i, opt in enumerate(optimizers):
                save_filename_opt.append('{0}_opt_{1}.pth'.format(epoch_label, i))
        save_path_net = os.path.join(self.save_dir, save_filename_net)
        torch.save(network.cpu().state_dict(), save_path_net)
        for i, opt in enumerate(optimizers):
            save_path_opt = os.path.join(self.save_dir, save_filename_opt[i])
            torch.save(opt.state_dict(), save_path_opt)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
        if isinstance(epoch_label, int):
            save_filename = '{0:03d}_net_{1}.pth'.format(epoch_label, network_label)
        else:
            save_filename = '{0}_net_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        # state_dict = torch.load(save_path)
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if 'module' in k:
        #         namelist = k.split('.')
        #         namelist.remove('module')
        #         name = '.'.join(namelist)  # remove `module.`
        #     else: name = k
        #     new_state_dict[name] = v
        # # load params
        # network.load_state_dict(new_state_dict)
        network.load_state_dict(torch.load(save_path))

    def load_optimizer(self, optimizer, network_label, epoch_label):
        print('Loading the optimizer {0} - epoch {1}'.format(network_label, epoch_label))
        if isinstance(epoch_label, int):
            save_filename = '{0:03d}_opt_{1}.pth'.format(epoch_label, network_label)
        else:
            save_filename = '{0}_opt_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        optimizer.load_state_dict(torch.load(save_path))

    def load_network_from_path(self, network, network_filepath, strict):
        network_label = os.path.basename(network_filepath)
        epoch_label = network_label.split('_')[0]
        print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
        network.load_state_dict(torch.load(network_filepath), strict=strict)

    # update learning rate (called once every epoch)
    def update_learning_rate(self, metric=None, epoch=None, policy=None):
        if policy=='polyLR':
            poly_lr_scheduler(self.optimizers, self.init_lr, epoch, lr_decay_iter=6, max_iter=1500)
        else:
            for scheduler in self.schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(metrics=metric)
                else:
                    scheduler.step()
                lr = self.optimizers[0].param_groups[0]['lr']
                print('current learning rate = %.7f' % lr)

    # returns the number of trainable parameters
    def get_number_parameters(self):
        return get_n_parameters(self.net)

    # clean up the GPU memory
    def destructor(self):
        del self.net
        del self.input
