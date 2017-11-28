import errno
import hashlib
import os
import sys
import tarfile
import h5py
import torch.utils.data as data
import torch
import numpy as np
import random
from PIL import Image
import cv2
from transforms.transforms import RandomAffine
import time
from dataset import MyDataset

class PascalVOC(MyDataset):

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split = 'train',
                 resize = False,
                 imsize = 256):

        CLASSES = ['<eos>','airplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'dining table', 'dog', 'horse',
                 'motorcycle', 'person', 'potted plant',
                 'sheep', 'sofa', 'train', 'tv']

        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.max_seq_len = args.gt_maxseqlen
        self.image_dir = os.path.join(args.pascal_dir, 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = args.batch_size
        if self.batch_size == 1:
            self.crop = False
        else:
            self.crop = True
        self.flip = augment
        if augment:
            self.augmentation_transform = RandomAffine(rotation_range=args.rotation,
                                                    translation_range=args.translation,
                                                    shear_range=args.shear,
                                                    zoom_range=(args.zoom,max(args.zoom*2,1.0)),
                                                    interp = 'nearest')

        else:
            self.augmentation_transform = None
        self.zoom = args.zoom
        self.augment = augment
        self.imsize = imsize
        self.resize = resize
        self.masks_dir = os.path.join(args.pascal_dir, 'ProcMasks')
        splits_dir = os.path.join(args.pascal_dir, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, split+'.txt')

        self.image_files = []
        with open(os.path.join(split_f), "r") as lines:
            for line in lines:
                self.image_files.append(line.rstrip('\n'))

    def get_raw_sample(self,index):
        """
        Returns sample data in raw format (no resize)
        """
        image_file = os.path.join(self.image_dir,self.image_files[index].rstrip() + '.jpg')
        img = Image.open(image_file).convert('RGB')

        mask = np.load(os.path.join(self.masks_dir,self.image_files[index].rstrip() + '.npy'))
        ins = mask[:,:,1]
        seg = mask[:,:,0]

        return img, ins, seg
