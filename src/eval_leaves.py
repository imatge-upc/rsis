import matplotlib
matplotlib.use('Agg')
from args import get_parser
from utils.utils import batch_to_var, make_dir, outs_perms_to_cpu, load_checkpoint
from modules.model import RIASS, FeatureExtractor
from test import test
import scipy.misc
from dataloader.dataset_utils import get_dataset
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import sys, os
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image


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

        dataset = get_dataset(args, self.split, image_transforms, augment=False, imsize=args.imsize)

        self.loader = data.DataLoader(dataset,batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             drop_last=False)

        self.sample_list = dataset.get_sample_list()
        self.args = args

        encoder_dict, decoder_dict, _, _, load_args = load_checkpoint(args.model_name)
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
            break

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
        results_root_dir = os.path.join('../models', args.model_name, args.model_name + '_results_test')
        make_dir(results_root_dir)
        results_dir = os.path.join(results_root_dir, 'A1')
        make_dir(results_dir)
        print "Creating annotations for leaves validation..."
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            x, y_mask, y_class, sw_mask, sw_class = batch_to_var(self.args, inputs, targets)
            out_masks, _, stop_probs = test(self.args, self.encoder, self.decoder, x)

            for sample in range(self.batch_size):
                sample_idx = self.sample_list[sample + acc_samples]
                image_dir = os.path.join(sample_idx.split('.')[0] + '.png')
                im = scipy.misc.imread(image_dir)
                h = im.shape[0]
                w = im.shape[1]

                mask_sample = np.zeros([h, w])
                sample_idx = sample_idx.split('/')[-1].split('.')[0]
                img_masks = out_masks[sample]

                instance_id = 0

                class_scores = stop_probs[sample]

                print('Start')
                print(image_dir)

                for time_step in range(self.T):
                    mask = img_masks[time_step].cpu().numpy()
                    mask = scipy.misc.imresize(mask, [h, w])

                    class_scores_mask = class_scores[time_step].cpu().numpy()
                    class_score = class_scores_mask[0]
                    if class_score > args.class_th:
                        mask_sample[mask > args.mask_th * 255] = time_step
                        instance_id += 1

                file_name = os.path.join(results_dir, sample_idx + '.png')
                file_name_prediction = file_name.replace('rgb.png', 'label.png')

                im = Image.fromarray(mask_sample).convert('L')
                im.save(file_name_prediction)

            acc_samples += self.batch_size


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    E = Evaluate(args)
    E.create_figures()
