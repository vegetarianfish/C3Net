import torchsample.transforms as ts
from pprint import pprint
import random
import math
import torch as th

class GHJRandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.
        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p
        v : boolean
            whether to vertically flip w/ probability p
        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, x, y=None, z=None):
        x = x.numpy()
        if y is not None:
            y = y.numpy()
        if z is not None:
            z = z.numpy()
        # horizontal flip with p = self.p
        if self.horizontal:
            if random.random() < self.p:
                x = x.swapaxes(2, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 2)
                if y is not None:
                    y = y.swapaxes(2, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 2)
                if z is not None:
                    z = z.swapaxes(2, 0)
                    z = z[::-1, ...]
                    z = z.swapaxes(0, 2)
        # vertical flip with p = self.p
        if self.vertical:
            if random.random() < self.p:
                x = x.swapaxes(1, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 1)
                if y is not None:
                    y = y.swapaxes(1, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 1)
                if z is not None:
                    z = z.swapaxes(1, 0)
                    z = z[::-1, ...]
                    z = z.swapaxes(0, 1)
        if y is None:
            # must copy because torch doesnt current support neg strides
            return th.from_numpy(x.copy())
        elif z is None:
            return th.from_numpy(x.copy()),th.from_numpy(y.copy())
        else:
            return th.from_numpy(x.copy()),th.from_numpy(y.copy()),th.from_numpy(z.copy())

class GHJSpecialCrop(object):

    def __init__(self, size, crop_type=0):
        """
        Perform a special crop - one of the four corners or center crop
        Arguments
        ---------
        size : tuple or list
            dimensions of the crop
        crop_type : integer in {0,1,2,3,4}
            0 = center crop
            1 = top left crop
            2 = top right crop
            3 = bottom right crop
            4 = bottom left crop
        """
        if crop_type not in {0, 1, 2, 3, 4}:
            raise ValueError('crop_type must be in {0, 1, 2, 3, 4}')
        self.size = size
        self.crop_type = crop_type
    
    def __call__(self, x, y=None, z=None):
        if self.crop_type == 0:
            # center crop
            x_diff  = (x.size(1)-self.size[0])/2.
            y_diff  = (x.size(2)-self.size[1])/2.
            ct_x    = [int(math.ceil(x_diff)),x.size(1)-int(math.floor(x_diff))]
            ct_y    = [int(math.ceil(y_diff)),x.size(2)-int(math.floor(y_diff))]
            indices = [ct_x,ct_y]        
        elif self.crop_type == 1:
            # top left crop
            tl_x = [0, self.size[0]]
            tl_y = [0, self.size[1]]
            indices = [tl_x,tl_y]
        elif self.crop_type == 2:
            # top right crop
            tr_x = [0, self.size[0]]
            tr_y = [x.size(2)-self.size[1], x.size(2)]
            indices = [tr_x,tr_y]
        elif self.crop_type == 3:
            # bottom right crop
            br_x = [x.size(1)-self.size[0],x.size(1)]
            br_y = [x.size(2)-self.size[1],x.size(2)]
            indices = [br_x,br_y]
        elif self.crop_type == 4:
            # bottom left crop
            bl_x = [x.size(1)-self.size[0], x.size(1)]
            bl_y = [0, self.size[1]]
            indices = [bl_x,bl_y]
        
        x = x[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]

        if z is not None:
            y = y[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]
            z = z[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]
            return x, y, z
        elif y is not None:
            y = y[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]
            return x, y
        else:
            return x

class Transformations:

    def __init__(self, name):
        self.name = name

        # Input patch and scale size
        self.scale_size = (192, 192, 1)
        self.patch_size = (128, 128, 1)
        # self.patch_size = (208, 272, 1)

        # Affine and Intensity Transformations
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.0

        # Divisibility factor for testing
        self.division_factor = (16, 16, 1)

    def get_transformation(self):
        return {
            'ukbb_sax': self.cmr_3d_sax_transform,
            'hms_sax':  self.hms_sax_transform,
            'test_sax': self.test_3d_sax_transform,
            'acdc_sax': self.cmr_3d_sax_transform,
            'us':       self.ultrasound_transform,
            'acdc_sax_edge': self.cmr_3d_sax_transform,
            'acdc_sax_edge_atlas': self.atlas_transform,
        }[self.name]()

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts):
        t_opts = getattr(opts, self.name)

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'scale_size'):       self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'):       self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'):        self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'):       self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'):        self.scale_val = t_opts.scale
        if hasattr(t_opts, 'inten_val'):        self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'division_factor'):  self.division_factor = t_opts.division_factor

    def ukbb_sax_transform(self):

        train_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),    # Pads a Numpy image to the given size
                                      ts.ToTensor(),        # Converts a numpy array to torch.Tensor
                                      ts.ChannelsFirst(),   # Transposes a tensor so that the channel dim is first.
                                      ts.TypeCast(['float', 'float']),  # Cast a torch.Tensor to a different type
                                      ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),   # Randomly flip an image horizontally and/or vertically with some probability.
                                      ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val, # Perform an affine transforms with various sub-transforms, using only one interpolation and without having to instantiate each sub-transform individually.
                                                      zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
                                      ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.RandomCrop(size=self.patch_size),
                                      ts.TypeCast(['float', 'long'])
                                ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.SpecialCrop(size=self.patch_size, crop_type=0),
                                      ts.TypeCast(['float', 'long'])
                                ])

        return {'train': train_transform, 'valid': valid_transform}

    def cmr_3d_sax_transform(self):

        train_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),
                                      ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                                      zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      # ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.RandomCrop(size=self.patch_size),
                                      ts.TypeCast(['float', 'long'])
                                ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      # ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.SpecialCrop(size=self.patch_size, crop_type=0),
                                      ts.TypeCast(['float', 'long'])
                                ])

        return {'train': train_transform, 'valid': valid_transform}

    def atlas_transform(self):

        train_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                    #   ts.TypeCast(['float', 'float', 'float']),
                                      GHJRandomFlip(h=True, v=True, p=self.random_flip_prob),
                                      ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                                      zoom_range=self.scale_val, interp=('bilinear', 'nearest', 'nearest')),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      # ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.RandomCrop(size=self.patch_size),
                                      ts.TypeCast(['float', 'long', 'long'])
                                ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                    #   ts.TypeCast(['float', 'float', 'float']),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      # ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      GHJSpecialCrop(size=self.patch_size, crop_type=0),
                                      ts.TypeCast(['float', 'long', 'long'])
                                ])

        return {'train': train_transform, 'valid': valid_transform}

    def cmr_3d_sax_edge_transform(self):

        train_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float', 'float', 'float']),
                                      ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),
                                      ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                                      zoom_range=self.scale_val, interp=('bilinear', 'nearest', 'bilinear', 'nearest')),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      # ts.NormalizeMedic(norm_flag=(True, False, True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.RandomCrop(size=self.patch_size),
                                      ts.TypeCast(['float', 'long', 'float', 'long'])
                                ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float', 'float', 'float']),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.NormalizeMedic(norm_flag=(True, False, True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.SpecialCrop(size=self.patch_size, crop_type=0),
                                      ts.TypeCast(['float', 'long', 'float', 'long'])
                                ])

        return {'train': train_transform, 'valid': valid_transform}

    def hms_sax_transform(self):

        # Training transformation
        # 2D Stack input - 3D High Resolution output segmentation

        train_transform = []
        valid_transform = []

        # First pad to a fixed size
        # Torch tensor
        # Channels first
        # Joint affine transformation
        # In-plane respiratory motion artefacts (translation and rotation)
        # Random Crop
        # Normalise the intensity range
        train_transform = ts.Compose([])

        return {'train': train_transform, 'valid': valid_transform}

    def test_3d_sax_transform(self):
        test_transform = ts.Compose([ts.PadFactorNumpy(factor=self.division_factor),
                                     ts.ToTensor(),
                                     ts.ChannelsFirst(),
                                     ts.TypeCast(['float']),
                                     #ts.NormalizeMedicPercentile(norm_flag=True),
                                     ts.NormalizeMedic(norm_flag=True),
                                     ts.ChannelsLast(),
                                     ts.AddChannel(axis=0),
                                     ])

        return {'test': test_transform}


    def ultrasound_transform(self):

        train_transform = ts.Compose([ts.ToTensor(),
                                      ts.TypeCast(['float']),
                                      ts.AddChannel(axis=0),
                                      ts.SpecialCrop(self.patch_size,0),
                                      ts.RandomFlip(h=True, v=False, p=self.random_flip_prob),
                                      ts.RandomAffine(rotation_range=self.rotate_val,
                                                      translation_range=self.shift_val,
                                                      zoom_range=self.scale_val,
                                                      interp=('bilinear')),
                                      ts.StdNormalize(),
                                ])

        valid_transform = ts.Compose([ts.ToTensor(),
                                      ts.TypeCast(['float']),
                                      ts.AddChannel(axis=0),
                                      ts.SpecialCrop(self.patch_size,0),
                                      ts.StdNormalize(),
                                ])

        return {'train': train_transform, 'valid': valid_transform}
