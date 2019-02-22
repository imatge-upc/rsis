from args import get_parser
from modules.model import RNNDecoder, FeatureExtractor
import torchvision.models as models
from utils.hungarian import match, softIoU, MaskedNLL
from utils.utils import batch_to_var, make_dir, init_visdom, check_parallel
from utils.utils import outs_perms_to_cpu, save_checkpoint, load_checkpoint
from dataloader.dataset_utils import get_dataset
from scipy.ndimage.measurements import center_of_mass
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as data
from utils.objectives import MaskedNLLLoss, softIoULoss, MaskedBCELoss, MaskedMSELoss
import time
import math
import os
import warnings
import sys
from PIL import Image
from utils.tb_logger import Visualizer
import pickle
import random
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
warnings.filterwarnings("ignore")

LN = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_dataloaders(args):
    loaders = {}

    # init dataloaders for training and validation
    for split in ['train', 'val']:
        batch_size = args.batch_size
        imsize = args.imsize
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_transforms = transforms.Compose([to_tensor, normalize])

        # dataset and loaders for training and validation splits
        dataset = get_dataset(args,
                              split=split,
                              image_transforms=image_transforms,
                              augment=args.augment and split == 'train',
                              imsize=imsize)

        loaders[split] = data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=True)
        class_names = dataset.get_classes()
    return loaders, class_names


def runIter(args, encoder, decoder, x, y_mask, y_class, sw_mask,
            crits, optim, mode='train', keep_gradients=True):
    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """
    mask_siou, class_crit, stop_xentropy = crits

    T = args.maxseqlen
    hidden = None
    out_masks, out_classes, out_stops = [], [], []

    feats = encoder(x, keep_gradients)

    # scores_aux has a high value in masked positions, and also in already picked positions
    scores_aux = torch.zeros(y_mask.size(0), args.maxseqlen).to(device) + (1-sw_mask.float())*(LN)
    perm_idxs = torch.ones(y_mask.size(0), args.maxseqlen).to(device)
    if args.curriculum_learning:
        T = min(args.maxseqlen, args.limit_seqlen_to)

    stop_next = False
    # loop over sequence length and get predictions
    feed_masks = torch.zeros(y_mask.size(0), x.size()[-2]*x.size()[-1]).to(device)
    for t in range(0, T):

        if stop_next:
            break
        # activate the stopping bool after the masking variable cointains 0s in all positions
        # this iteration will run, but not the next
        if sw_mask[:, t].sum().float().cpu().data.numpy() == 0:
            stop_next = True

        out_mask, out_class, out_stop, hidden = decoder(feats, hidden,
                                                        feed_masks.view(y_mask.size(0), 1,
                                                                        x.size()[-2],
                                                                        x.size()[-1]))

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)

        # repeat predicted mask as many times as elements in ground truth.
        # to compute iou against all ground truth elements at once
        y_pred_i = out_mask.unsqueeze(0)
        y_pred_i = y_pred_i.permute(1, 0, 2)
        y_pred_i = y_pred_i.repeat(1, y_mask.size(1), 1)
        y_pred_i = y_pred_i.view(y_mask.size(0)*y_mask.size(1), y_mask.size(2))
        y_true_p = y_mask.view(y_mask.size(0)*y_mask.size(1), y_mask.size(2))

        c = softIoU(y_true_p, torch.sigmoid(y_pred_i))
        c = c.view(sw_mask.size(0), -1)

        c = scores_aux + c
        val, idx = torch.min(c, dim=-1)
        for b in range(y_mask.size(0)):
            scores_aux[b, idx[b]] = LN
        perm_idxs[:, t] = idx

        if args.use_feedback:
            if args.teacher_forcing:
                # pick ground truth mask of the picked idx (the one with lowest loss)
                for b in range(y_mask.size(0)):
                    feed_masks[b] = y_mask[b, idx[b], :].squeeze()
            else:
                feed_masks = (torch.sigmoid(out_mask) > 0.5).float().detach()

        # get predictions in list to concat later
        out_masks.append(out_mask)
        out_classes.append(out_class)
        out_stops.append(out_stop)

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks, 1).view(out_mask.size(0), len(out_masks), -1)
    out_classes = torch.cat(out_classes, 1).view(out_class.size(0), len(out_classes), -1)
    out_stops = torch.cat(out_stops, 1).view(out_stop.size(0), len(out_stops), -1)

    y_mask_perm, y_class_perm = y_mask, y_class
    for b in range(y_mask.size(0)):
        y_mask_perm[b, :, :] = y_mask[b, perm_idxs[b].long(), :]
        y_class_perm[b, :] = y_class[b, perm_idxs[b].long()]

    y_mask_perm = y_mask_perm[:, 0:t].contiguous()
    y_class_perm = y_class_perm[:, 0:t].contiguous()
    sw_mask = sw_mask[:, 0:t].contiguous().float()

    out_classes = out_classes.contiguous()
    out_masks = out_masks.contiguous()

    # assign weight proportional to the amount of samples with nonmasked elements in ith positions.
    weight_mask = sw_mask.size(0) / (sw_mask.sum(dim=0).unsqueeze(0) + 1e-6) * sw_mask

    loss_class = class_crit(y_class_perm.view(-1, 1), out_classes.view(-1, out_classes.size()[-1]))
    loss_class = torch.sum(loss_class.view(sw_mask.size()) * sw_mask) / torch.sum(sw_mask)

    loss_bce = stop_xentropy(y_mask_perm.view(-1, y_mask.size()[-1]),
                             out_masks.view(-1, out_masks.size()[-1]))

    loss_bce = torch.sum(loss_bce.view(sw_mask.size())*sw_mask) / torch.sum(sw_mask)

    out_masks = torch.sigmoid(out_masks)
    th_out_masks = (out_masks > 0.5).float()
    loss_mask_iou = mask_siou(y_mask_perm.view(-1, y_mask.size()[-1]),
                              out_masks.view(-1, out_masks.size()[-1]))

    loss_mask_iou = torch.sum(loss_mask_iou.view(sw_mask.size()) * weight_mask) / torch.sum(sw_mask)

    loss_mask_iou_th = mask_siou(y_mask_perm.view(-1, y_mask.size()[-1]),
                                 th_out_masks.view(-1, th_out_masks.size()[-1]))

    loss_mask_iou_th = torch.sum(loss_mask_iou_th.view(sw_mask.size()) * sw_mask) / torch.sum(sw_mask)

    loss_stop = stop_xentropy(sw_mask.float().view(-1, 1), out_stops.squeeze().view(-1, 1)).mean()

    # total loss is the weighted sum of all terms
    loss = args.iou_weight * loss_mask_iou

    if args.use_class_loss:
        loss += args.class_weight*loss_class
    if args.use_stop_loss:
        loss += args.stop_weight*loss_stop
    loss+=args.bce_weight*loss_bce

    optim.zero_grad()

    if mode == 'train':
        loss.backward()
        optim.step()

    losses = {'loss': loss.item(), 'iou': loss_mask_iou.item(), 'stop': loss_stop.item(),
              'class': loss_class.item(), 'bce': loss_bce.item(),
              'iou_th': loss_mask_iou_th.item()}

    return losses


def trainIters(args):

    args.curr_epoch = 0
    batch_size = args.batch_size
    model_dir = os.path.join('../models/', args.model_name)

    if args.tensorboard:
        tb_logs = '../models/tb_logs'
        make_dir(model_dir)
        tb_logs = os.path.join(tb_logs, args.model_name)
        make_dir(model_dir)
        logger = Visualizer(tb_logs, name='visual_results')

    if args.resume:
        # will resume training the model with name args.model_name
        encoder_dict, decoder_dict, opt_dict, args = load_checkpoint(args.model_name, args.use_gpu)

        encoder = FeatureExtractor(args)
        decoder = RNNDecoder(args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)
        args.batch_size = batch_size

    elif args.transfer_from != '':

        encoder_dict, decoder_dict, opt_dict, load_args = load_checkpoint(args.transfer_from, args.use_gpu)
        load_args.dropout_cls = args.dropout_cls
        encoder = FeatureExtractor(load_args)
        decoder = RNNDecoder(load_args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

    else:
        encoder = FeatureExtractor(args)
        decoder = RNNDecoder(args)

    # model checkpoints will be saved here
    make_dir(model_dir)

    # save parameters for future use
    pickle.dump(args, open(os.path.join(model_dir,'args.pkl'),'wb'))

    # params_cnn = list(encoder.base.parameters())
    if args.finetune_layers == 3:
        params_cnn = list(encoder.base.layer4.parameters()) + list(encoder.base.layer3.parameters()) \
                     + list(encoder.base.layer2.parameters())
    else:
        params_cnn = list(encoder.base.parameters())

    params = list(decoder.parameters()) + list(encoder.conv_embed.parameters())

    if args.finetune_after != -1 and args.finetune_after <= args.curr_epoch:
        print ('Fine tune CNN')
        keep_cnn_gradients = True
        optimizer = torch.optim.Adam([{'params': params}, {'params': params_cnn, 'lr': args.lr_cnn}], lr=args.lr)
    else:
        print ('Frozen CNN')
        optimizer = torch.optim.Adam(params, lr=args.lr)
        keep_cnn_gradients = False

    if args.resume:
        optimizer.load_state_dict(opt_dict)
        from collections import defaultdict
        optimizer.state = defaultdict(dict, optimizer.state)

    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(model_dir, 'train.log'))
        sys.stdout = open(os.path.join(model_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(model_dir, 'train.err'), 'w')

    print (args)
    # objective functions for mask and class outputs.
    # these return the average across samples in batch whose value
    # needs to be considered (those where sw is 1)
    # mask_xentropy = BalancedStableMaskedBCELoss()
    mask_siou = softIoULoss()

    class_xentropy = MaskedNLLLoss(balance_weight=None, gamma=args.gamma)
    stop_xentropy = MaskedBCELoss(mask_mode=False, gamma=args.gamma)

    if torch.cuda.device_count() > 1:
        decoder = torch.nn.DataParallel(decoder)
        encoder = torch.nn.DataParallel(encoder)
        mask_siou = torch.nn.DataParallel(mask_siou)
        class_xentropy = torch.nn.DataParallel(class_xentropy)
        stop_xentropy = torch.nn.DataParallel(stop_xentropy)
    if args.use_gpu:
        encoder.to(device)
        decoder.to(device)
        class_xentropy.to(device)
        mask_siou.to(device)
        stop_xentropy.to(device)

    crits = [mask_siou, class_xentropy, stop_xentropy]
    if args.use_gpu:
        torch.cuda.synchronize()
    start = time.time()

    # vars for early stopping
    best_val_loss = args.best_val_loss
    acc_patience = 0

    if args.curriculum_learning and epoch_resume == 0:
            args.limit_seqlen_to = 2

    # keep track of the number of batches in each epoch for continuity when plotting curves
    loaders, class_names = init_dataloaders(args)
    curr_epoch = args.curr_epoch
    for e in range(curr_epoch, args.max_epoch):
        args.curr_epoch = e
        print ("Epoch", e)

        if args.ss_after != -1 and args.ss_after <= e:
            args.teacher_forcing = False
            best_val_loss = 1000
            acc_patience = 0
        else:
            args.teacher_forcing = True
        print (args.teacher_forcing)

        # check if it's time to do some changes here
        if args.finetune_after != -1 and args.finetune_after <= e and not keep_cnn_gradients:
            print ('Starting to fine tune CNN')
            keep_cnn_gradients = True
            optimizer = torch.optim.Adam([{'params': params}, {'params': params_cnn, 'lr': args.lr_cnn}], lr=args.lr)
            acc_patience = 0
        if e >= args.class_loss_after and not args.use_class_loss and not args.class_loss_after == -1:
            print("Starting to learn class loss")
            args.use_class_loss = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
            acc_patience = 0
        if e >= args.stop_loss_after and not args.use_stop_loss and not args.stop_loss_after == -1:
            if args.curriculum_learning:
                if args.limit_seqlen_to > args.min_steps:
                    print("Starting to learn stop loss")
                    args.use_stop_loss = True
                    best_val_loss = 1000 # reset because adding a loss term will increase the total value
                    acc_patience = 0
            else:
                print("Starting to learn stop loss")
                args.use_stop_loss = True
                best_val_loss = 1000 # reset because adding a loss term will increase the total value
                acc_patience = 0

        # we validate after each epoch
        for split in ['train', 'val']:
            if split == 'train':
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()

            epoch_losses = {'loss': [], 'iou': [], 'iou_th':[], 'class': [], 'stop': [], 'bce': []}
            total_step = len(loaders[split])
            for batch_idx, (img, masks, cats, sw) in enumerate(loaders[split]):
                # send batch to GPU
                y_mask = masks.to(device)
                y_class = cats.to(device)
                x = img.to(device)
                sw_mask = sw.to(device)
                # we forward (and backward & update if training set)
                losses = runIter(args, encoder, decoder, x, y_mask,
                                 y_class, sw_mask,
                                 crits, optimizer, mode=split, keep_gradients=keep_cnn_gradients)

                for k, v in losses.items():
                    # store loss values in dictionary separately
                    epoch_losses[k].append(v)

                # print and display in visdom after some iterations
                if (batch_idx + 1) % args.print_every == 0:

                    te = time.time() - start
                    lossesstr = ''
                    if args.tensorboard:
                        logger.scalar_summary(mode=split + '_iter', epoch=e*total_step + batch_idx,
                                              **{k: np.mean(v[-args.print_every:]) for k, v in epoch_losses.items() if
                                                 v})
                        # logger.histo_summary(model=decoder, step=e*total_step + batch_idx)
                    for k in epoch_losses.keys():
                        if len(epoch_losses[k]) == 0:
                            continue
                        this_one = "%s: %.4f" % (k, np.mean(epoch_losses[k][-args.print_every:]))
                        lossesstr += this_one + ', '
                        # this only displays nll loss on captions, the rest of losses will be in tensorboard logs
                    strtoprint = 'Split: %s, Epoch [%d/%d], Step [%d/%d], Losses: %sTime: %.4f' % (split, e,
                                                                                                   args.max_epoch,
                                                                                                   batch_idx,
                                                                                                   total_step,
                                                                                                   lossesstr,
                                                                                                   te)
                    print(strtoprint)
                    torch.cuda.synchronize()
                    start = time.time()

            if args.tensorboard:
                logger.scalar_summary(mode=split, epoch=e, **{k: np.mean(v) for k, v in epoch_losses.items() if v})

            # compute mean val losses within epoch

        mt = np.mean(epoch_losses['iou_th'])

        if mt < best_val_loss:
            print ("Saving checkpoint.")
            best_val_loss = mt
            args.best_val_loss = best_val_loss
            # saves model, params, and optimizers
            save_checkpoint(args, encoder, decoder, optimizer)
            acc_patience = 0
        else:
            acc_patience += 1

        if acc_patience > args.patience and not args.use_class_loss and not args.class_loss_after == -1:
            print("Starting to learn class loss")
            acc_patience = 0
            args.use_class_loss = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
        if acc_patience > args.patience and args.curriculum_learning and args.limit_seqlen_to < args.maxseqlen:
            print("Adding one step more:")
            acc_patience = 0
            args.limit_seqlen_to += args.steps_cl
            print(args.limit_seqlen_to)
            best_val_loss = 1000

        if acc_patience > args.patience and not keep_cnn_gradients and not args.finetune_after == -1:
            print("Starting to update encoder")
            acc_patience = 0
            keep_cnn_gradients = True
            optimizer = torch.optim.Adam([{'params': params}, {'params': params_cnn, 'lr': args.lr_cnn}], lr=args.lr)

        if acc_patience > args.patience and not args.use_stop_loss and not args.stop_loss_after == -1:
            if args.curriculum_learning:
                print("Starting to learn stop loss")
                if args.limit_seqlen_to > args.min_steps:
                    acc_patience = 0
                    args.use_stop_loss = True
                    best_val_loss = 1000 # reset because adding a loss term will increase the total value
            else:
                print("Starting to learn stop loss")
                acc_patience = 0
                args.use_stop_loss = True
                best_val_loss = 1000 # reset because adding a loss term will increase the total value

        # early stopping after N epochs without improvement
        if acc_patience > args.patience_stop:
            break

    if args.tensorboard:
        logger.close()


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    #np.random.seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    trainIters(args)
