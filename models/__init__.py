# Abstract level model definition
# Returns the model class for specified network type
import os
from .feedforward_seg_model import FeedForwardSegmentation

class ModelOpts:
    def __init__(self):
        self.gpu_ids = [0]
        self.isTrain = True
        self.continue_train = False
        self.which_epoch = int(0)
        self.save_dir = './checkpoints/default'
        self.model_type = 'unet'
        self.input_nc = 1
        self.output_nc = 4
        self.lr_rate = 1e-12
        self.l2_reg_weight = 0.0
        self.weight_decay = 0.0
        self.feature_scale = 4
        self.tensor_dim = '2D'
        self.path_pre_trained_model = None
        self.criterion = 'cross_entropy'
        self.type = 'seg'
        self.res_connect = False
        self.dropout = 0

        # Attention
        self.nonlocal_mode = 'concatenation'
        self.attention_dsample = (2,2,2)

        # Attention Classifier
        self.aggregation_mode = 'concatenation'


    def initialise(self, json_opts):
        opts = json_opts

        self.raw = json_opts
        # self.gpu_ids = opts.gpu_ids
        self.isTrain = opts.isTrain
        self.save_dir = os.path.join(opts.checkpoints_dir, opts.experiment_name)
        self.model_type = opts.model_type
        self.input_nc = opts.input_nc
        self.output_nc = opts.output_nc
        self.continue_train = opts.continue_train
        self.which_epoch = opts.which_epoch

        if hasattr(opts, 'type'): self.type = opts.type
        if hasattr(opts, 'l2_reg_weight'): self.l2_reg_weight = opts.l2_reg_weight
        if hasattr(opts, 'weight_decay'): self.weight_decay = opts.weight_decay
        if hasattr(opts, 'optim'): self.optim = opts.optim
        if hasattr(opts, 'lr_rate'):       self.lr_rate = opts.lr_rate
        if hasattr(opts, 'feature_scale'): self.feature_scale = opts.feature_scale
        if hasattr(opts, 'tensor_dim'):    self.tensor_dim = opts.tensor_dim

        if hasattr(opts, 'path_pre_trained_model'): self.path_pre_trained_model = opts.path_pre_trained_model
        if hasattr(opts, 'criterion'):              self.criterion = opts.criterion

        if hasattr(opts, 'nonlocal_mode'): self.nonlocal_mode = opts.nonlocal_mode
        if hasattr(opts, 'attention_dsample'): self.attention_dsample = opts.attention_dsample
        # Classifier
        if hasattr(opts, 'aggregation_mode'): self.aggregation_mode = opts.aggregation_mode
        if hasattr(opts, 'res_connect'): self.res_connect = opts.res_connect
        if hasattr(opts, 'dropout'): self.dropout = opts.dropout
        if hasattr(opts, 'ignore_index'): self.ignore_index = None if opts.void_classes == [] else opts.ignore_index
        if hasattr(opts, 'void_classes'): self.void_classes = None if opts.void_classes == [] else opts.void_classes
        if hasattr(opts, 'edge_weight'): self.edge_weight = opts.edge_weight
        if hasattr(opts, 'dice_weight'): self.dice_weight = opts.dice_weight
        self.aspp_channel = opts.aspp_channel if hasattr(opts, 'aspp_channel') else 0
        self.edge_type = opts.edge_type if hasattr(opts, 'edge_type') else None
        self.edge_input = opts.edge_input if hasattr(opts, 'edge_input') else True
        self.edge_att_weight = opts.edge_att_weight if hasattr(opts, 'edge_att_weight') else 0
        self.seg_grad_weight = opts.seg_grad_weight if hasattr(opts, 'seg_grad_weight') else 0

def get_model(json_opts):

    # Neural Network Model Initialisation
    model = None
    model_opts = ModelOpts()
    model_opts.initialise(json_opts)

    # Print the model type
    print('\nInitialising model {}'.format(model_opts.model_type))

    model_type = model_opts.type
    assert model_type == 'seg'

    model = FeedForwardSegmentation()


    # Initialise the created model
    model.initialize(model_opts)
    print("Model [%s] is created" % (model.name()))

    return model
