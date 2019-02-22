import errno
import hashlib
import os
import sys
import tarfile
import h5py
import torch.utils.data as data
import torch
from torchvision import transforms
from .dataset_utils import scale, flip_crop
import numpy as np
from PIL import Image
import time
from scipy.ndimage.filters import gaussian_filter


class MyDataset(data.Dataset):

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split = 'train',
                 resize = False,
                 imsize = 256):

        self.max_seq_len = args.gt_maxseqlen
        self.classes = []
        self.imsize = imsize
        self.augment = augment

    def get_classes(self):
        return self.classes

    def __getitem__(self, index):
        data = self.get_raw_sample(index)
        img = data['image']
        if self.transform is not None:
            img = self.transform(img)

        ins = torch.from_numpy(data['masks'])
        if self.augmentation_transform is not None:
            img, ins = self.augmentation_transform(img, ins)

        sw = np.array(data['cats'] != 0).astype(float)
        # print (img.size(), ins.size(), sw.size())
        return img, ins.view(ins.size(0), ins.size(1)*ins.size(2)).float(), torch.from_numpy(data['cats']).long(), \
               torch.from_numpy(sw)

    def __len__(self):
        return len(self.image_files)

    def get_sample_list(self):
        return self.image_files
