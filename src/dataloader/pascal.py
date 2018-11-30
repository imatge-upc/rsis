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
from .transforms.transforms import RandomAffine
import time
from .dataset import MyDataset
import lmdb
import pickle


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
        self.pascal_dir = args.pascal_dir
        self.batch_size = args.batch_size
        if augment:
            self.augmentation_transform = RandomAffine(rotation_range=args.rotation,
                                                    translation_range=args.translation,
                                                    shear_range=args.shear,
                                                    interp = 'nearest')

        else:
            self.augmentation_transform = None
        self.zoom = args.zoom
        self.imsize = imsize
        self.resize = resize
        self.masks_dir = os.path.join(args.pascal_dir, 'ProcMasks')
        self.lmdb = args.lmdb
        splits_dir = os.path.join(args.pascal_dir, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, split+'.txt')

        self.image_files = []
        with open(os.path.join(split_f), "r") as lines:
            for line in lines:
                self.image_files.append(line.rstrip('\n'))

        self.lmdb_file = lmdb.open(os.path.join(args.pascal_dir, 'lmdbs', 'lmdb_' + split), max_readers=1,
                                   readonly=True, lock=False, readahead=False, meminit=False)

    def get_raw_sample(self,index):
        """
        Returns sample data in raw format (no resize)
        """
        name = self.image_files[index].rstrip()
        data = {}
        
        if self.lmdb:
          with self.lmdb_file.begin(write=False) as txn:
  
              for suff in ['image', 'masks', 'cats', 'boxes']:
                  datum = txn.get((name + '_' + suff).encode())
                  if suff == 'image':
                      datum = np.fromstring(datum, dtype=np.uint8)
                      datum = np.reshape(datum, (self.imsize, self.imsize, 3))
                      datum = Image.fromarray(datum.astype('uint8'), 'RGB')
                  elif suff == 'masks':
                      datum = np.fromstring(datum)
                      datum = np.reshape(datum, (10, self.imsize, self.imsize))
                      datum = datum[0:self.max_seq_len]
                      # datum = np.transpose(datum, (1, 2, 0))
                  else:
                      datum = np.fromstring(datum)
                      datum = datum[0:self.max_seq_len]
  
                  data[suff] = datum
        else:
            data = pickle.load(open(os.path.join(self.pascal_dir, 'lmdbs', name + '.pkl'),'rb'))
            data['masks'] = np.reshape(data['masks'][0:self.max_seq_len], (self.max_seq_len, self.imsize, self.imsize))
            data['cats'] = data['cats'][0:self.max_seq_len].squeeze()
            data['boxes'] = data['boxes'][0:self.max_seq_len].squeeze()
            data['image'] = Image.fromarray(data['image'].astype('uint8'), 'RGB')
        return data