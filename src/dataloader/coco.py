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
import pycocotools.mask as mask
from pycocotools.coco import COCO

class MSCOCO(MyDataset):

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split = 'train',
                 sseg = False,
                 resize = False,
                 imsize = 256):

        self.classes = ['<eos>','person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter',
                'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle',
                'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.num_classes = len(self.classes)
        self.class2ind = {}

        for i,c in enumerate(self.classes):
            self.class2ind[c] = i
        if split == 'val' or split =='test':
            self.imagesplit = 'val'
        else:
            self.imagesplit = 'train'
        root = os.path.join(args.coco_dir,'images',self.imagesplit+'2014')

        if split == 'val':
            annsplit = 'minival'
        elif split == 'train':
            annsplit = 'train'
        else:
            annsplit = 'valminusminival'
        annFile = '%s/annotations/instances_%s2014.json'%(args.coco_dir,annsplit)

        self.coco = COCO(annFile)
        self.image_files = list(self.coco.imgs.keys())

        self.zoom = args.zoom
        self.augment = augment
        self.imsize = imsize
        self.resize = resize
        self.root = root

        self.transform = transform
        self.target_transform = target_transform
        self.max_seq_len = args.gt_maxseqlen
        self.predict_eos_mask = args.predict_eos_mask
        self.sseg = sseg
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
                                                    zoom_range=(args.zoom,1),
                                                    interp = 'nearest')

        else:
            self.augmentation_transform = None


    def get_raw_sample(self, index):
        coco = self.coco
        img_id = self.image_files[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        anns = [ann for ann in anns if "area" in ann and type(ann['segmentation']) == list]
        anns = sorted(anns, key=lambda ann: -ann["area"])

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        h = np.shape(img)[0]
        w = np.shape(img)[1]
        ins = np.zeros((h,w))
        seg = np.zeros((h,w))

        num_obj = 1
        for ann in anns:
            class_name = self.coco.loadCats(ann["category_id"])[0]["name"]
            class_id = self.class2ind[class_name]
            m = mask.decode(mask.frPyObjects(ann['segmentation'], h,w))
            # mask sometimes can be separated in different connected components.
            # these are treated as they belong to the same object (cause they do)
            m = np.max(m,axis=-1).squeeze()
            ins[m==1] = num_obj
            seg[m==1] = class_id
            num_obj+=1

        return img, ins, seg

    def imname_from_idx(self,idx):
        return self.coco.loadImgs(idx)[0]['file_name']
    def image_split(self):
        return self.imagesplit
