import numpy as np
from scipy.ndimage.interpolation import zoom
import torch
from .transforms.transforms import random_crop
import random

def get_dataset(args, split, image_transforms = None, augment = False,imsize = 256):

    if args.dataset =='pascal':
        from .pascal import PascalVOC as MyChosenDataset
    elif args.dataset == 'cityscapes':
        from .cityscapes import CityScapes as MyChosenDataset
    elif args.dataset == 'leaves':
        from .leaves import LeavesDataset as MyChosenDataset


    dataset = MyChosenDataset(args,
                            split = split,
                            transform = image_transforms,
                            target_transform = None,
                            augment = augment,
                            resize = args.resize,
                            imsize = imsize)
    return dataset


def scale(img,ins,seg):
    h = img.size(1)
    w = img.size(2)

    # have masks be of shape (1,h,w)
    seg = np.expand_dims(seg, axis=-1)
    ins = np.expand_dims(ins, axis=-1)
    ins = resize_(ins, h, w)
    seg = resize_(seg, h, w)
    seg = seg.squeeze()
    ins = ins.squeeze()

    return ins, seg

def flip_crop(img,ins,seg,flip=True,crop=True,imsize=256):
    h = img.size(1)
    w = img.size(2)
    seg = np.expand_dims(seg, axis=0)
    ins = np.expand_dims(ins, axis=0)

    if random.random() < 0.5 and flip:
        img = np.flip(img.numpy(),axis=2).copy()
        img = torch.from_numpy(img)
        ins = np.flip(ins,axis=2).copy()
        seg = np.flip(seg,axis=2).copy()

    ins = torch.from_numpy(ins)
    seg = torch.from_numpy(seg)
    if crop:
        img, ins, seg = random_crop([img,ins,seg],(imsize,imsize), (h,w))
    return img, ins, seg


def pascal_palette():

    # RGB to int conversion

    palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20,
             (224,  224, 192) : 255 }

    return palette



def sequence_palette():

    # RGB to int conversion

    palette = {(  0,   0,   0) : 0 ,
             (0,   255,   0) : 1 ,
             (  255, 0,   0) : 2 ,
             (0, 0,   255) : 3 ,
             (  255,   0, 255) : 4 ,
             (0,   255, 255) : 5 ,
             (  255, 128, 0) : 6 ,
             (102, 0, 102) : 7 ,
             ( 51,   153,   255) : 8 ,
             (153,   153,   255) : 9 ,
             ( 153, 153,   0) : 10,
             (178, 102,   255) : 11,
             ( 204,   0, 204) : 12,
             (0,   102, 0) : 13,
             ( 102, 0, 0) : 14,
             (51, 0, 0) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20,
             (224,  224, 192) : 21 }

    return palette

def convert_from_color_segmentation(arr_3d, image_height, image_width):

    # 3D image to 2D map with int values
    palette = pascal_palette()

    reshape_array = np.reshape(arr_3d, [image_height * image_width, 3])

    #still too slow!!
    arr_2d = np.fromiter([palette.get((x[0], x[1], x[2]), 0) for x in reshape_array],
                         reshape_array.dtype)

    return np.reshape(np.asarray(arr_2d), arr_3d.shape[0:2])

def resize_(img, height, width):
    '''
    Resize a 3D array (image) to the size specified in parameters
    '''
    zoom_h = float(height) / img.shape[0]
    zoom_w = float(width) / img.shape[1]
    img = zoom(img, [zoom_h, zoom_w, 1], mode='nearest', order=0)
    return img
