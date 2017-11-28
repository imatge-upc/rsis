import os
import numpy as np
from PIL import Image
from transforms.transforms import RandomAffine
from dataset import MyDataset
import glob


class LeavesDataset(MyDataset):

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split = 'train',
                 resize = False,
                 imsize = 256):

        CLASSES = ['<eos>', 'leaf']

        self.split = split
        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.max_seq_len = args.gt_maxseqlen
        self.image_dir = os.path.join(args.leaves_dir, 'A1')
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = args.batch_size
        self.no_run_coco_eval = True
        if self.batch_size == 1:
            self.crop = False
        else:
            self.crop = True
        self.flip = augment
        if augment and not resize:
            self.augmentation_transform = RandomAffine(rotation_range=args.rotation,
                                                    translation_range=args.translation,
                                                    shear_range=args.shear,
                                                    interp = 'nearest')
        elif augment and resize:
            self.augmentation_transform = RandomAffine(rotation_range=args.rotation,
                                                    translation_range=args.translation,
                                                    shear_range=args.shear,
                                                    zoom_range=(args.zoom,1),
                                                    interp = 'nearest')

        else:
            self.augmentation_transform = None

        self.zoom = args.zoom
        self.augment = augment
        self.imsize = imsize
        self.resize = resize

        self.image_files = []
        self.gt_files = []
        self.train_image_files = []
        self.train_gt_files = []
        self.val_image_files = []
        self.val_gt_files = []
        self.test_image_files = []

        self.image_total_files = glob.glob(os.path.join(args.leaves_dir, '*_rgb.png'))
        self.gt_total_files = [w.replace('_rgb', '_label') for w in self.image_total_files]

        self.test_image_total_files = glob.glob(os.path.join(args.leaves_test_dir, '*_rgb.png'))

        """
        we split the training set between validation and training. The first 96 images will be for training and
        the other will be for validation
        """
        for i in range(len(self.image_total_files)):
            if i < 96:
                self.train_image_files.append(self.image_total_files[i])
                self.train_gt_files.append(self.gt_total_files[i])
            else:
                self.val_image_files.append(self.image_total_files[i])
                self.val_gt_files.append(self.gt_total_files[i])

        for j in range(len(self.test_image_total_files)):
            self.test_image_files.append(self.test_image_total_files[j])

        if split == "train":
            self.image_files = self.train_image_files
            self.gt_files = self.train_gt_files
        elif split == "val":
            self.image_files = self.val_image_files
            self.gt_files = self.val_gt_files
        elif split == "test":
            self.image_files = self.test_image_files

    def get_raw_sample(self,index):
        """
        Returns sample data in raw format (no resize)
        """
        image_file = os.path.join(self.image_dir, self.image_files[index])
        img = Image.open(image_file).convert('RGB')

        if self.split != "test":
            gt_file = os.path.join(self.image_dir, self.gt_files[index])
            gt = np.array(Image.open(gt_file))

            ins = gt.copy()
            seg = gt.copy()
            seg[seg > 0] = 1

            return img, ins, seg

        else:

            img_fake = np.array(img)[:, :, 0]
            return img, img_fake, img_fake
