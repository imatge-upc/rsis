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
import cv2
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

    def get_raw_sample(self,index):
        """
        Returns sample data in raw format (no resize)
        """
        img = []
        ins = []
        seg = []

        return img, ins, seg

    def __getitem__(self, index):
        img, ins, seg = self.get_raw_sample(index)
        # image will be resized to square (if self.resize is true)
        if self.resize:
            image_resize = transforms.Scale((self.imsize,self.imsize), Image.BILINEAR)
        else:
            image_resize = transforms.Scale(self.imsize, Image.BILINEAR)

        img = image_resize(img)
        '''
        img = np.array(img)
        img = img.astype(np.float32)[:, :, ::-1]  # RGB to BGR!!!
        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img.copy())
        '''
        if self.transform is not None:
            img = self.transform(img)

        ins, seg = scale(img, ins, seg)
        img, ins, seg = flip_crop(img, ins, seg, flip=self.flip,crop=self.crop,imsize=self.imsize)
        ins = ins.float()
        seg = seg.float()

        if self.augmentation_transform is not None:
            img, ins, seg = self.augmentation_transform(img, ins, seg)
        # back to numpy to extract separate instances from transformed mask arrays
        seg = seg.numpy().squeeze()
        ins = ins.numpy().squeeze()

        target, boxes = self.sequence_from_masks(ins,seg)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, boxes

    def __len__(self):
        return len(self.image_files)

    def get_sample_list(self):
        return self.image_files

    def sequence_from_masks(self, ins, seg):
        """
        Reads segmentation masks and outputs sequence of binary masks and labels
        """

        num_classes = len(self.classes)
        instance_ids = np.unique(ins)[1:]

        h = ins.shape[0]
        w = ins.shape[1]

        total_num_instances = len(instance_ids)
        num_instances = max(self.max_seq_len, total_num_instances)

        gt_classes = np.zeros((num_instances, 1))
        gt_seg = np.zeros((num_instances, ins.shape[0]*ins.shape[1]))
        box_mask = np.zeros((num_instances, 2, (ins.shape[0]//8)*(ins.shape[1]//8)), dtype=float)
        size_masks = np.zeros((num_instances, )) # for sorting by size
        sample_weights_mask = np.zeros((num_instances, 1))
        sample_weights_class = np.zeros((num_instances, 1))

        for i in range(total_num_instances):

            id_instance = instance_ids[i]

            # id of the class the instance belongs to
            # translates from dataset_id (corresponding to pascal) to class_id
            # (corresponding to our class id)
            unique_class_ids = np.unique(seg[ins == id_instance])
            dataset_class_id = unique_class_ids[0]
            class_id = dataset_class_id
            gt_classes[i] = class_id

            # binary mask
            aux_mask = np.zeros((h, w))
            aux_mask[ins == id_instance] = 1

            y, x = np.where(aux_mask == 1)
            y_min, x_min, y_max, x_max = min(y)//8, min(x)//8, max(y)//8, max(x)//8
            x1_mask = np.zeros((h//8, w//8), dtype=float)
            x1_mask[y_min, x_min] = 1

            x2_mask = np.zeros((h//8, w//8), dtype=float)
            x2_mask[y_max, x_max] = 1

            sig = 0.04 * w//8
            x1_mask_gauss = gaussian_filter(x1_mask, sigma=sig)
            x2_mask_gauss = gaussian_filter(x2_mask, sigma=sig)
            box_mask[i, 0, :] = np.reshape(x1_mask_gauss, h*w//(8*8))
            box_mask[i, 1, :] = np.reshape(x2_mask_gauss, h*w//(8*8))

            gt_seg[i, :] = np.reshape(aux_mask, h*w)
            size_masks[i] = np.sum(gt_seg[i, :])
            num_instances = num_instances + 1
            sample_weights_mask[i] = 1
            sample_weights_class[i] = 1

        # objects sorted by size
        idx_sort = np.argsort(size_masks)[::-1]

        # After sorting, take only the first N instances for training
        gt_classes = gt_classes[idx_sort][:self.max_seq_len]
        gt_seg = gt_seg[idx_sort][:self.max_seq_len]
        box_mask = box_mask[idx_sort][:self.max_seq_len]
        sample_weights_mask = sample_weights_mask[idx_sort][:self.max_seq_len]
        sample_weights_class = sample_weights_class[idx_sort][:self.max_seq_len]

        # put the end of sequence token if it happens before max_num_instances
        if self.max_seq_len > total_num_instances:

            gt_classes[total_num_instances:] = 0
            gt_seg[total_num_instances:, :] = 0
            box_mask[total_num_instances:] = 0
            sample_weights_class[total_num_instances] = 1
        targets = np.concatenate((gt_seg, gt_classes), axis=1)
        targets = np.concatenate((targets, sample_weights_mask), axis=1)
        targets = np.concatenate((targets, sample_weights_class), axis=1)

        # box_mask = np.transpose(box_mask, (1, 0, 2))

        return targets, box_mask
