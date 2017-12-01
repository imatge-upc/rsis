import os
import numpy as np
from PIL import Image
from transforms.transforms import RandomAffine
from dataset import MyDataset
import glob


class CityScapes(MyDataset):

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split='train',
                 resize=False,
                 imsize = 256):

        CLASSES = ['<eos>', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.max_seq_len = args.gt_maxseqlen
        self.image_files = glob.glob(os.path.join(args.cityscapes_dir, 'leftImg8bit', split, '*', '*.png'))
        self.ins_files = [w.replace('/leftImg8bit/', '/gtFine/') for w in self.image_files]
        self.ins_files = [w.replace('_leftImg8bit.png', '_gtFine_instanceIds.png') for w in self.ins_files]
        self.seg_files = [w.replace('/leftImg8bit/', '/gtFine/') for w in self.image_files]
        self.seg_files = [w.replace('_leftImg8bit.png', '_gtFine_labelIds.png') for w in self.seg_files]

        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = args.batch_size
        self.no_run_coco_eval = True

        self.crop = args.crop
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

    def get_raw_sample(self, index):
        """
        Returns sample data in raw format (no resize)
        """

        image_file = os.path.join(self.image_files[index])
        ins_file = os.path.join(self.ins_files[index])

        img = Image.open(image_file).convert('RGB')
        ins = np.array(Image.open(ins_file))
        seg = ins.copy()/1000

        # don't train for caravan and trailer
        seg[seg == 29] = 0
        seg[seg == 30] = 0

        # change ids, so that they start by id=1 
        seg[seg > 0] -= 23

        # change ids of 3 last classes, considering that we are not training for caravan and trailer 
        seg[seg == 8] = 6
        seg[seg == 9] = 7
        seg[seg == 10] = 8

        seg_aux = seg.copy()
        seg_aux[seg_aux > 0] = 1

        # mask the instance map with those classes that are not being trained
        ins = ins*seg_aux
        ins[ins < 24000] = 0
        unique_ids = np.unique(ins)

        # associate an individual id for each instance
        for i in range(len(unique_ids)):
            ins[ins == unique_ids[i]] = i

        return img, ins, seg
