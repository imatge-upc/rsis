import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_parser
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask 
from utils.utils import batch_to_var, make_dir, outs_perms_to_cpu, load_checkpoint
from scipy.ndimage.measurements import center_of_mass
from modules.model import RIASS, FeatureExtractor
from test import test
from PIL import Image
import scipy.misc
from dataloader.dataset_utils import get_dataset
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import pickle
import sys, os
import json
from scipy.ndimage.interpolation import zoom
from collections import OrderedDict
from skimage import measure
from collections import Counter


class Evaluate():

    def __init__(self,args):
        self.split = args.eval_split
        self.display = args.display
        self.dataset = args.dataset
        self.all_classes = args.all_classes
        self.T = args.maxseqlen
        self.batch_size = args.batch_size

        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if args.use_coco_weights:
            # when using coco weights transformations are handled within loader
            image_transforms = None
        else:
            image_transforms = transforms.Compose([to_tensor,normalize])

        dataset = get_dataset(args, self.split, image_transforms,augment=False)

        self.loader = data.DataLoader(dataset,batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False)

        self.sample_list = dataset.get_sample_list()
        self.args = args

        encoder_dict, decoder_dict, _, _, load_args = load_checkpoint(args.model_name)
        load_args.seg_checkpoint_path = args.seg_checkpoint_path
        self.args.use_feedback = load_args.use_feedback
        self.args.base_model = load_args.base_model
        self.hidden_size = load_args.hidden_size
        self.args.nconvlstm = load_args.nconvlstm
        self.encoder = FeatureExtractor(load_args)
        self.decoder = RIASS(load_args)

        if args.ngpus > 1 and args.use_gpu:
            self.decoder = torch.nn.DataParallel(self.decoder,device_ids=range(args.ngpus))
            self.encoder = torch.nn.DataParallel(self.encoder,device_ids=range(args.ngpus))

        # check if the model was trained using multiple gpus
        trained_parallel = False
        for k, v in encoder_dict.items():
            if k[:7] == "module.":
                trained_parallel = True
            break;

        if trained_parallel and not args.ngpus > 1:
            # create new OrderedDict that does not contain "module."
            new_encoder_state_dict = OrderedDict()
            new_decoder_state_dict = OrderedDict()
            for k, v in encoder_dict.items():
                name = k[7:] # remove "module."
                new_encoder_state_dict[name] = v
            for k, v in decoder_dict.items():
                name = k[7:] # remove "module."
                new_decoder_state_dict[name] = v
            encoder_dict = new_encoder_state_dict
            decoder_dict = new_decoder_state_dict

        self.encoder.load_state_dict(encoder_dict)
        self.decoder.load_state_dict(decoder_dict)

        if args.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()

    def create_figures(self):

        acc_samples = 0
        results_dir = os.path.join('../models', args.model_name, args.model_name + '_results')

        make_dir(results_dir)
        masks_dir = os.path.join(args.model_name + '_masks')
        abs_masks_dir = os.path.join(results_dir, masks_dir)
        make_dir(abs_masks_dir)
        print "Creating annotations for cityscapes validation..."
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            x, y_mask, y_class, sw_mask, sw_class = batch_to_var(self.args, inputs, targets)
            outs, true_perms, stop_probs = test(self.args, self.encoder, self.decoder, x, y_mask, y_class, sw_mask, sw_class)

            class_ids = [24, 25, 26, 27, 28, 31, 32, 33]

            for sample in range(self.batch_size):

                sample_idx = self.sample_list[sample + acc_samples]
                image_dir = os.path.join(sample_idx.split('.')[0] + '.png')
                im = scipy.misc.imread(image_dir)
                h = im.shape[0]
                w = im.shape[1]

                sample_idx = sample_idx.split('/')[-1].split('.')[0]

                results_file = open(os.path.join(results_dir, sample_idx + '.txt'), 'w')
                img_masks = outs[0][sample]

                instance_id = 0

                class_scores = outs[1][sample]
                stop_scores = stop_probs[sample]

                for time_step in range(self.T):
                    mask = img_masks[time_step].cpu().numpy()
                    mask = (mask > args.mask_th)

                    if args.keep_biggest_blob:
                    
                        h_mask = mask.shape[0]
                        w_mask = mask.shape[1]
                    
                        mask = (mask > 0)
                        labeled_blobs = measure.label(mask, background=0).flatten()
                    
                        # find the biggest one
                        count = Counter(labeled_blobs)
                        s = []
                        max_num = 0
                        for v, k in count.iteritems():
                            if v == 0:
                                continue
                            if k > max_num:
                                max_num = k
                                max_label = v
                        # build mask from the largest connected component
                        segmentation = (labeled_blobs == max_label).astype("uint8")
                        mask = segmentation.reshape([h_mask, w_mask]) * 255

                    mask = scipy.misc.imresize(mask, [h, w])
                    class_scores_mask = class_scores[time_step].cpu().numpy()
                    stop_scores_mask = stop_scores[time_step].cpu().numpy()
                    class_score = np.argmax(class_scores_mask)

                    for i in range(len(class_scores_mask) - 1):
                        name_instance = sample_idx + '_' + str(instance_id) + '.png'
                        pred_class_score = class_scores_mask[i+1]
                        objectness = stop_scores_mask[0]
                        pred_class_score *= objectness
                        scipy.misc.imsave(os.path.join(abs_masks_dir, name_instance), mask)
                        results_file.write(masks_dir + '/' + name_instance + ' ' + str(class_ids[i]) + ' ' + str(pred_class_score) + '\n')
                        instance_id += 1

                results_file.close()

            acc_samples += self.batch_size


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    E = Evaluate(args)
    E.create_figures()

