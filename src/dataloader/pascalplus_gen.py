from scipy.io import loadmat
import os
import numpy as np
from scipy.misc import imsave
from tqdm import *
import argparse
from distutils.dir_util import copy_tree
from dataset_utils import pascal_palette
import random
random.seed(1337)

'''
Creates a training dataset composed of VOC 2012 augmented
with the one presented in:

Hariharan et al. Semantic contours from inverse detectors. ICCV 2011.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--contours_dir',
                    default = '../../data/dettention/datasets/benchmark_RELEASE/dataset/')
parser.add_argument('--voc_dir', default = '../../data/dettention/datasets/VOC2012/')
parser.add_argument('--vocplus_dir', default = '../../data/dettention/datasets/VOCAug2/', help='This folder will be created')
parser.add_argument('--val_split', default = 0.10, type=int,help='Percentage of samples to use for validation')
parser.add_argument('--force_gen', dest='force_gen', action='store_true')
parser.add_argument('--nocopy', dest='copy', action='store_false')
parser.set_defaults(force_gen=False)
parser.set_defaults(copy=True)

args = parser.parse_args()

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def read_file(filedir):
    with open(filedir,'r') as f:
        lines = f.readlines()
    return lines

def write_file(filedir,itemlist):
    with open(filedir,'w') as f:
        for item in itemlist:
            f.write(item)

palette = pascal_palette()
id_to_rgb = {v:k for k,v in palette.items()}
root = args.contours_dir

# create directories for new dataset
make_dir(args.vocplus_dir)
make_dir(os.path.join(args.vocplus_dir,'SegmentationClass'))
make_dir(os.path.join(args.vocplus_dir,'SegmentationObject'))
make_dir(os.path.join(args.vocplus_dir,'ImageSets'))
make_dir(os.path.join(args.vocplus_dir,'JPEGImages'))
make_dir(os.path.join(args.vocplus_dir,'ImageSets','Segmentation'))

# train and val splits from augmented dataset will both be used for training
for split in ['train','val']:
    print ("Processing %s set:"%(split))
    names = read_file(os.path.join(args.contours_dir,split+'.txt'))
    print ("Found %d images to process."%(len(names)))
    for n, name in tqdm(enumerate(names)):
        # extract segmentation info from mat files
        seg_png = os.path.join(args.vocplus_dir,'SegmentationClass',name.rstrip()+'.png')
        obj_png = os.path.join(args.vocplus_dir,'SegmentationObject',name.rstrip()+'.png')
        if not os.path.isfile(seg_png) or not os.path.isfile(obj_png) or args.force_gen:
            matfile = loadmat(os.path.join(args.contours_dir,'inst',name.rstrip()+'.mat'))
            seg_object = matfile['GTinst']['Segmentation'][0][0]
            matfile = loadmat(os.path.join(args.contours_dir,'cls',name.rstrip()+'.mat'))
            classes = matfile['GTcls']['Segmentation'][0][0]
            sem_seg = np.zeros((seg_object.shape[0],seg_object.shape[1],3),dtype=np.uint8)
            ins_seg = np.zeros((seg_object.shape[0],seg_object.shape[1],3),dtype=np.uint8)
            for i in np.unique(seg_object):
                if i == 0:
                    continue
                class_ins = classes[i-1][0]
                # encode class with corresponding RGB triplet
                sem_seg[seg_object == i,0] = id_to_rgb[class_ins][0]
                sem_seg[seg_object == i,1] = id_to_rgb[class_ins][1]
                sem_seg[seg_object == i,2] = id_to_rgb[class_ins][2]

                # use i as class id (it will be unique) to color code instance masks
                ins_seg[seg_object == i,0] = id_to_rgb[i][0]
                ins_seg[seg_object == i,1] = id_to_rgb[i][1]
                ins_seg[seg_object == i,2] = id_to_rgb[i][2]

            imsave(seg_png, sem_seg)
            imsave(obj_png, ins_seg)
        else:
            print "File %d already exists ! Skipping..."%(n)

print "Merging index lists and creating trainval split..."
voc_train = read_file(os.path.join(args.voc_dir,'ImageSets','Segmentation','train.txt'))
print "VOC 2012 - train: Found %d samples"%(len(voc_train))
contours_train = read_file(os.path.join(args.contours_dir,'train.txt'))
print "Contours - train: Found %d samples"%(len(contours_train))
contours_val = read_file(os.path.join(args.contours_dir,'val.txt'))
print "Contours - val: Found %d samples"%(len(contours_val))

# the validation set of pascal voc will be used for testing
test_samples = read_file(os.path.join(args.voc_dir,'ImageSets','Segmentation','val.txt'))
print "Val set from VOC will be used for testing."

samples = []
samples.extend(voc_train)
# Make sure we don't train with samples in the split we use for testing !!
for sample in contours_train:
    if sample not in test_samples:
        samples.append(sample)
for sample in contours_val:
    if sample not in test_samples:
        samples.append(sample)

samples = list(set(samples))
print "Total samples for training: ", len(samples)
print "Note that images of VOC are part of the Contours dataset (duplicates are only used once)"
random.shuffle(samples)

sep = int(len(samples)*(1-args.val_split))
train_samples = samples[:sep]
val_samples = samples[sep:]

print "The percentage of samples used for val was set to be %.2f"%(args.val_split)
print "Training samples:", len(train_samples)
print "Validation samples:", len(val_samples)

write_file(os.path.join(args.vocplus_dir,'ImageSets','Segmentation','train.txt'),train_samples)
write_file(os.path.join(args.vocplus_dir,'ImageSets','Segmentation','val.txt'),val_samples)
write_file(os.path.join(args.vocplus_dir,'ImageSets','Segmentation','test.txt'),test_samples)

if args.copy:
    print "Copying images from Contours dataset..."
    copy_tree(os.path.join(args.contours_dir,'img'),
              os.path.join(args.vocplus_dir,'JPEGImages'))

    print ("Copying files from Pascal VOC to new dataset directory...")
    copy_tree(os.path.join(args.voc_dir,'SegmentationClass'),
              os.path.join(args.vocplus_dir,'SegmentationClass'))
    copy_tree(os.path.join(args.voc_dir,'SegmentationObject'),
              os.path.join(args.vocplus_dir,'SegmentationObject'))
    copy_tree(os.path.join(args.voc_dir,'JPEGImages'),
              os.path.join(args.vocplus_dir,'JPEGImages'))

print "All done."
