import pickle
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom
import argparse
import lmdb
from torchvision import transforms
import pickle


MAX_SIZE = 1e12


def load_and_resize_image(impath, desired_size):

    rm, bm, gm = int(0.485*255), int(0.456*255), int(0.406*255)

    im = Image.open(impath)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size), (rm, bm, gm))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))

    h = new_size[0]
    w = new_size[1]

    return new_im, h, w

def load_and_resize_masks(maskpath, square_size, resize_dims, maxseqlen):

    mask = np.load(maskpath)
    h, w = resize_dims
    zoom_h = float(h) / mask.shape[0]
    zoom_w = float(w) / mask.shape[1]
    mask = zoom(mask, [zoom_h, zoom_w, 1], mode='nearest', order=0)

    new_mask = np.zeros((square_size, square_size, 2))
    h_paste = (square_size - h) // 2
    w_paste = (square_size - w) // 2

    new_mask[h_paste:h_paste+h, w_paste:w_paste+w] = mask

    ins = new_mask[:, :, 1]
    seg = new_mask[:, :, 0]

    instance_ids = np.unique(ins)[1:]
    total_num_instances = len(instance_ids)
    num_instances = max(maxseqlen, total_num_instances)

    gt_classes = np.zeros((num_instances, 1))
    gt_seg = np.zeros((num_instances, ins.shape[0] * ins.shape[1]))
    size_masks = np.zeros((num_instances,))  # for sorting by size
    boxes = np.zeros((num_instances, 4))

    for i in range(total_num_instances):

        id_instance = instance_ids[i]
        unique_class_ids = np.unique(seg[ins == id_instance])
        dataset_class_id = unique_class_ids[0]
        class_id = dataset_class_id
        gt_classes[i] = class_id

        # binary mask
        aux_mask = np.zeros((square_size, square_size))
        aux_mask[ins == id_instance] = 1

        y, x = np.where(aux_mask == 1)
        y_min, x_min, y_max, x_max = min(y), min(x), max(y), max(x)
        boxes[i] = np.array([y_min, x_min, y_max, x_max])

        gt_seg[i, :] = np.reshape(aux_mask, square_size * square_size)
        size_masks[i] = np.sum(gt_seg[i, :])
        num_instances = num_instances + 1

    # objects sorted by size
    idx_sort = np.argsort(size_masks)[::-1]

    # After sorting, take only the first N instances for training
    gt_classes = gt_classes[idx_sort][:maxseqlen]
    gt_seg = gt_seg[idx_sort][:maxseqlen]
    boxes = boxes[idx_sort][:maxseqlen]

    return gt_seg, gt_classes, boxes


def main(args):

    parts = {}

    image_dir = os.path.join(args.root, 'JPEGImages')
    splits_dir = os.path.join(args.root, 'ImageSets/Segmentation')
    masks_dir = os.path.join(args.root, 'ProcMasks')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for split in ['train', 'val', 'test']:

        parts[split] = lmdb.open(os.path.join(args.save_dir, 'lmdb_' + split), map_size=int(MAX_SIZE))
        split_f = os.path.join(splits_dir, split + '.txt')

        image_files = []
        with open(os.path.join(split_f), "r") as lines:
            for line in lines:
                image_files.append(line.rstrip('\n'))

        for index in tqdm(range(len(image_files))):
            image_file = os.path.join(image_dir, image_files[index].rstrip() + '.jpg')
            mask_file = os.path.join(masks_dir, image_files[index].rstrip() + '.npy')

            img, h, w = load_and_resize_image(image_file, args.imsize)
            masks, categories, boxes = load_and_resize_masks(mask_file, args.imsize, (h,w), args.maxseqlen)

            img = np.array(img).astype(np.uint8)
            name = image_files[index].rstrip()
            
            if args.lmdb:
                with parts[split].begin(write=True) as txn:
                    txn.put((name + '_image').encode(), img)
                    txn.put((name + '_masks').encode(), masks)
                    txn.put((name + '_cats').encode(), categories)
                    txn.put((name + '_boxes').encode(), boxes)
            else:   
                data = {'image':img, 'masks':masks, 'cats':categories, 'boxes':boxes}
                with open(os.path.join(args.save_dir, name + '.pkl'), 'wb') as f:
                    pickle.dump(data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/asalvador/datasets/VOCAug/')
    parser.add_argument('--save_dir', type=str, default='/home/asalvador/datasets/VOCAug/lmdbs/')
    parser.add_argument('--imsize', type=int, default=256)
    parser.add_argument('--maxseqlen', type=int, default=10)
    parser.add_argument('--lmdb', dest='lmdb', action='store_true')
    parser.set_defaults(lmdb=False)
    args = parser.parse_args()

    main(args)