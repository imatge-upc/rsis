import os
import argparse
import numpy as np
import dataset_utils as ut
from scipy.misc import imread, imresize
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask
from tqdm import *
import json
import pickle

CLASSES = ['<eos>','airplane', 'bicycle', 'bird', 'boat',
         'bottle', 'bus', 'car', 'cat', 'chair',
         'cow', 'dining table', 'dog', 'horse',
         'motorcycle', 'person', 'potted plant',
         'sheep', 'sofa', 'train', 'tv']

def create_annotation(imname, gt_mask, class_id, score,crowd):
    """
    Creates annotation object following the COCO API ground truth format
    """
    cat_name = CLASSES[class_id]
    ann = dict()
    ann['image_id'] = imname.rstrip()
    ann['category_id'] = class_id
    ann['category_name'] = cat_name
    segmentation = (gt_mask > 0.5).astype("uint8")
    height = segmentation.shape[0]
    width = segmentation.shape[1]
    ann['segmentation'] = mask.encode(np.asfortranarray(segmentation.reshape([height,width,1])))[0]
    ann['score'] = score
    #ann['iscrowd'] = crowd
    ann['ignore'] = crowd
    return ann

def precompute(image_name, data_dir,ignore_id=255):
    idx = image_name.rstrip()
    segmentation_dir = os.path.join(data_dir, 'SegmentationClass', idx + '.png')
    instance_dir = os.path.join(data_dir, 'SegmentationObject', idx + '.png')
    seg_ = imread(segmentation_dir)
    ins_ = imread(instance_dir)
    image_height = seg_.shape[0]
    image_width = seg_.shape[1]
    seg_ = ut.convert_from_color_segmentation(seg_,image_height,image_width)
    ins_ = ut.convert_from_color_segmentation(ins_,image_height,image_width)

    ignore_mask =np.zeros((image_height,image_width))
    ignore_mask[seg_ == ignore_id] = 1
    ins_[seg_ == ignore_id] = 0
    seg_[seg_ == ignore_id] = 0

    seg_ = np.reshape(seg_,(image_height, image_width,1))
    ins_ = np.reshape(ins_,(image_height, image_width,1))

    masks = np.concatenate((seg_,ins_),axis=-1)

    if len(np.unique(ignore_mask)) == 0:
        ignore_mask = None
    return masks, ignore_mask

def get_imnames(args,split):
    names = []
    splits_dir = os.path.join(args.pascal_dir, 'ImageSets/Segmentation')
    split_f = os.path.join(splits_dir, split+'.txt')
    with open(split_f, "r") as lines:
        for line in lines:
            names.append(line.rstrip())
    return names

def make_coco(name, masks, ignore_mask):

    seg = masks[:,:,0]
    ins = masks[:,:,1]

    instance_ids = np.unique(ins)[1:]

    h = ins.shape[0]
    w = ins.shape[1]

    total_num_instances = len(instance_ids)

    sample_anns = list()
    for i in range(total_num_instances):

        id_instance = instance_ids[i]
        unique_class_ids = np.unique(seg[ins == id_instance])
        dataset_class_id = unique_class_ids[0]
        class_id = dataset_class_id

        # binary mask
        gt_mask = np.zeros((h, w))
        gt_mask[ins==id_instance] = 1

        sample_anns.append(create_annotation(name, gt_mask, class_id, score=1,crowd=0))

    if ignore_mask is not None:
        for i in range(1,len(CLASSES)):
            ignored = create_annotation(name, ignore_mask, i, score=1,crowd=1)
            sample_anns.append(ignored)

    return sample_anns

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pascal_dir',default='/work/asalvador/dev/data/dettention/datasets/VOCPlus2/')
    parser.add_argument('--split',default='train')
    parser.add_argument('--forcegen', dest='forcegen', action='store_true')
    parser.set_defaults(forcegen=False)
    args = parser.parse_args()

    split = args.split
    save_dir = os.path.join(args.pascal_dir,'ProcMasks')
    make_dir(save_dir)

    names = get_imnames(args,split)
    N = len(names)

    gt_annotations = list()

    for i,name in tqdm(enumerate(names)):
        file_to_save = os.path.join(save_dir,name+'.npy')
        if not os.path.isfile(file_to_save) or args.forcegen:
            masks,ignore_mask = precompute(name,args.pascal_dir)
            np.save(file_to_save,masks)
        else:
            print "Found masks for sample %s. Skipping."%(name)
        masks = np.load(file_to_save)
        gt_annotations.extend(make_coco(name,masks,ignore_mask))

    print "Saving COCO-like file for Pascal evaluation."
    file_to_save = os.path.join(args.pascal_dir,'VOCGT_%s.pkl'%(split))
    with open(file_to_save, 'wb') as outfile:
        pickle.dump(gt_annotations, outfile)
    print "Done."
