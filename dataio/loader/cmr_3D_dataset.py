import torch
import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join
from .utils import load_nifti_img, check_exceptions, is_image_file
from models.layers.loss import one_hot, class2one_hot
from scipy.ndimage import distance_transform_edt as distance
import SimpleITK as sitk
from operator import itemgetter
import numpy as np
import csv
import json

# np.random.seed(123)

FLAG_EDGE_NO_BG = False


class CMR3DDataset(data.Dataset):
    def __init__(self, root_dir, split, fold_no, transform=None, preload_data=False, ignore_index=255, void_classes=None,
                 edge=False, edge_input=True, edge_type=None, boundary=False, HU=None, norm_std=0.0, norm_mean=0.0):
        super(CMR3DDataset, self).__init__()
        print("root_dir:", root_dir)
        data_name = root_dir.split('/')[-1]
        print("Dataset: ", data_name)
        print("Fold: ", fold_no)
        image_dir = join(root_dir, 'image')
        target_dir = join(root_dir, 'label')
        self.boundary = boundary    
        self.edge = edge            
        self.edge_type = edge_type 
        self.edge_input = edge_input
        if edge:
            self.edge_class, self.edge_position = edge_type.split('_')

        img_filenames, tgt_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]), \
                                       sorted([join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])

        with open(join(root_dir, 'split_5_fold.json'), 'r+') as f:
            split_dict_read = json.loads(f.read())
        val_list = split_dict_read['val_%d'%fold_no]
        trainidx = split_dict_read['train_%d'%fold_no]
        val_list = list(np.array(val_list)-1)
        trainidx = list(np.array(trainidx)-1)
        fold = {}
        fold[fold_no] = val_list
        
        testidx = fold[fold_no]
        trainidx = list(set(trainidx) - set(testidx))
        if split=='train':
            self.image_filenames  = itemgetter(*trainidx)(img_filenames)
            self.target_filenames = itemgetter(*trainidx)(tgt_filenames)
            print("Images for training: ", self.image_filenames)
        elif split=='validation':
            self.image_filenames  = itemgetter(*testidx)(img_filenames)
            self.target_filenames = itemgetter(*testidx)(tgt_filenames)
            print("Images for validation: ", self.image_filenames)
            if not isinstance(self.image_filenames, tuple):
                self.image_filenames  = [self.image_filenames]
                self.target_filenames = [self.target_filenames]

        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))


        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        self.ignore_index = ignore_index
        self.void_classes = void_classes
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [load_nifti_img(ii, dtype=np.float32)[0] for ii in self.image_filenames]
            self.raw_labels = [load_nifti_img(ii, dtype=np.uint8)[0] for ii in self.target_filenames]
            self.raw_metas = [load_nifti_img(ii, dtype=np.float32)[1] for ii in self.image_filenames]
            print('raw images and labels are loaded')
            if HU:
                self.raw_images = [np.clip(ii, HU[0], HU[1]) for ii in self.raw_images]
            if norm_mean and norm_std:
                self.raw_images = [(ii - norm_mean)/norm_std for ii in self.raw_images]

            unique_labels = np.unique(self.raw_labels[0]).tolist()
            if self.void_classes is not None:
                self.target_labels = list(set(unique_labels) - set(self.void_classes))
                print('Void label(s): ', self.void_classes)
            else:
                self.target_labels = unique_labels
            self.raw_labels = [self.__targetremap__(target) for target in self.raw_labels]

            print('Target remap is done: ')
            print(np.unique(self.raw_labels[0]))


        else:
            raise NotImplementedError



    def __targetremap__(self, target):
        # ignore first and then map
        if self.void_classes is not None:
            for vc in self.void_classes:
                target[target==vc] = self.ignore_index
        for i in range(len(self.target_labels)):
            target[target==self.target_labels[i]] = i

        return target

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            raise NotImplementedError
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.float32)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
            target = self.__targetremap__(target)
            return input, target
        else:
            input = np.copy(self.raw_images[index])
            size = input.shape
            target = np.copy(self.raw_labels[index])

            if self.edge and not self.boundary:
                if self.transform:
                    input, target = self.transform(input, target) 
                    target_i = target.clone().numpy()
                    edge_target = per_sample_edge_generator(target_i, self.edge_position)
                    if FLAG_EDGE_NO_BG:
                        edge_target -= 1
                        edge_target[edge_target == -1] = 0
                    edge_target = torch.from_numpy(edge_target).long()

                    if not self.edge_input: return input, (target, edge_target), size
                    raise NotImplementedError
            else:
                if self.transform:
                    input, target = self.transform(input, target)
                if self.boundary:
                    dist = torch.tensor(self.dist_maps[index], dtype=torch.float32)
                    return input, (target, dist)
                return input, target, size

    def __len__(self):
        return len(self.image_filenames)


def binary_edge_generate(label_i, class_id, type='inner'):
    label = label_i.copy()
    label[label != class_id] = 0
    label[label == class_id] = 1
    label_sum = label \
                + np.roll(label, shift=1, axis=0) \
                + np.roll(label, shift=-1, axis=0) \
                + np.roll(label, shift=1, axis=1) \
                + np.roll(label, shift=-1, axis=1) \
                + np.roll(label, shift=1, axis=2) \
                + np.roll(label, shift=-1, axis=2)
    if type == 'inner':
        surface = np.where((label_sum > 0) & (label_sum < 7) & (label == 1), class_id, 0)
    elif type == 'outer':
        surface = np.where((label_sum > 0) & (label_sum < 7) & (label == 0), class_id, 0)
    elif type == 'regular':
        surface = np.where((label_sum > 0) & (label_sum < 7), class_id, 0)
    else: raise NotImplementedError
    return surface

def per_sample_edge_generator(label_i, type):
    edge = np.zeros(label_i.shape, dtype=np.uint8)
    unique_labels = np.unique(label_i).tolist()
    for j in unique_labels:
        if j == 0:
            pass
        else:
            tmp = binary_edge_generate(label_i, j, type)
            edge = np.maximum(tmp, edge).astype(np.uint8)
    return edge

def unison_shuffled_copies(a, b):
    a, b = np.array(a), np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p].tolist(), b[p].tolist()