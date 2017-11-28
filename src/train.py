from args import get_parser
from modules.model import RSIS, FeatureExtractor
import torchvision.models as models
from utils.hungarian import match, softIoU, MaskedNLL
from utils.utils import get_optimizer, batch_to_var, make_dir, init_visdom, check_parallel
from utils.utils import outs_perms_to_cpu, save_checkpoint, load_checkpoint, get_base_params,get_skip_params,merge_params
from dataloader.dataset_utils import get_dataset
from scipy.ndimage.measurements import center_of_mass
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as data
from utils.objectives import MaskedNLLLoss, softIoULoss, MaskedBCELoss
import time
import math
import os
import warnings
import sys
from PIL import Image
import pickle
import random

warnings.filterwarnings("ignore")

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
                              imsize = imsize)

        loaders[split] = data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=True)
        class_names = dataset.get_classes()
    return loaders, dataset


def runIter(args, encoder, decoder, x, y_mask, y_class, sw_mask,
            sw_class, crits, optims, mode='train'):
    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """
    mask_siou, class_crit, stop_xentropy = crits
    # mask_siou, class_crit, stop_xentropy = crits

    enc_opt, dec_opt = optims
    T = args.maxseqlen
    hidden = None
    loss_mask_iou = 0
    loss_class = 0
    out_masks = []
    out_classes = []
    out_stops = []
    if mode == 'train':
        encoder.train(True)
        decoder.train(True)
    else:
        encoder.train(False)
        decoder.train(False)
    feats = encoder(x)
    scores = torch.ones(y_mask.size(0),args.gt_maxseqlen,args.maxseqlen)

    if args.curriculum_learning:
        T = min(args.maxseqlen, args.limit_seqlen_to)

    stop_next = False
    # loop over sequence length and get predictions
    for t in range(0, T):

        if stop_next:
            break
        # activate the stopping bool after the masking variable cointains 0s in all positions
        # this iteration will run, but not the next
        if sw_mask[:, t].sum().float().cpu().data.numpy() == 0:
            stop_next = True

        out_mask, out_class, out_stop, hidden = decoder(feats, hidden)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)

        # repeat predicted mask as many times as elements in ground truth.
        # to compute iou against all ground truth elements at once
        y_pred_i = out_mask.unsqueeze(0)
        y_pred_i = y_pred_i.permute(1,0,2)
        y_pred_i = y_pred_i.repeat(1,y_mask.size(1),1)
        y_pred_i = y_pred_i.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))
        y_true_p = y_mask.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))

        c = args.iou_weight * softIoU(y_true_p, y_pred_i)
        c = c.view(sw_mask.size(0),-1)
        scores[:,:,t] = c.cpu().data

        # get predictions in list to concat later
        out_masks.append(out_mask)
        out_classes.append(out_class)
        out_stops.append(out_stop)

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_mask.size(0),len(out_masks), -1)
    out_classes = torch.cat(out_classes,1).view(out_class.size(0),len(out_classes), -1)
    out_stops = torch.cat(out_stops,1).view(out_stop.size(0),len(out_stops), -1)

    # get permutations of ground truth based on predictions (CPU computation)
    masks = [y_mask,out_masks]
    classes = [y_class,out_classes]

    sw_mask_mult = sw_mask.unsqueeze(-1).repeat(1,1,args.maxseqlen).byte()
    sw_mask_mult_T = sw_mask[:,0:args.maxseqlen].unsqueeze(-1).repeat(1,1,args.gt_maxseqlen).byte()
    sw_mask_mult_T = sw_mask_mult_T.permute(0,2,1).byte()
    sw_mask_mult = (sw_mask_mult.data.cpu() & sw_mask_mult_T.data.cpu()).float()
    scores = torch.mul(scores,sw_mask_mult) + (1-sw_mask_mult)*10

    scores = Variable(scores,requires_grad=False)
    if args.use_gpu:
        scores = scores.cuda()

    y_mask_perm, y_class_perm, _ = match(masks, classes, scores)

    # move permuted ground truths back to GPU
    y_mask_perm = Variable(torch.from_numpy(y_mask_perm[:,0:t]), requires_grad=False)
    y_class_perm = Variable(torch.from_numpy(y_class_perm[:,0:t]), requires_grad=False)

    if args.use_gpu:
        y_mask_perm = y_mask_perm.cuda()
        y_class_perm = y_class_perm.cuda()

    sw_mask = Variable(torch.from_numpy(sw_mask.data.cpu().numpy()[:,0:t])).contiguous()
    sw_class = Variable(torch.from_numpy(sw_class.data.cpu().numpy()[:,0:t])).contiguous()

    if args.use_gpu:
        sw_mask = sw_mask.float().cuda()
        sw_class = sw_class.float().cuda()

    # all losses are masked with sw_mask except the stopping one, which has one extra position
    loss_class = class_crit(y_class_perm.view(-1,1),out_classes.view(-1,out_classes.size()[-1]), sw_mask)

    loss_class = torch.mean(loss_class)
    loss_mask_iou = mask_siou(y_mask_perm.view(-1,y_mask_perm.size()[-1]),out_masks.view(-1,out_masks.size()[-1]), sw_mask.view(-1,1))
    loss_mask_iou = torch.mean(loss_mask_iou)

    # stopping loss is computed using the masking variable as ground truth
    #loss_stop = stop_xentropy(sw_mask.view(-1,1).float(),out_stops.view(-1,out_stops.size()[-1]), sw_class)
    loss_stop = stop_xentropy(sw_mask.float(),out_stops.squeeze(), sw_class.view(-1,1))
    loss_stop = torch.mean(loss_stop)

    # total loss is the weighted sum of all terms
    loss = args.iou_weight * loss_mask_iou

    if args.use_class_loss:
        loss+=args.class_weight*loss_class
    if args.use_stop_loss:
        loss+=args.stop_weight*loss_stop

    enc_opt.zero_grad()
    dec_opt.zero_grad()
    decoder.zero_grad()
    encoder.zero_grad()

    if mode == 'train':
        loss.backward()
        dec_opt.step()
        if args.update_encoder:
            enc_opt.step()

    losses = [loss.data[0], loss_mask_iou.data[0], loss_stop.data[0], loss_class.data[0]]

    out_masks = torch.sigmoid(out_masks)
    outs = [out_masks.data, out_classes.data]

    perms = [y_mask_perm.data, y_class_perm.data]
    del loss, loss_mask_iou, loss_stop, loss_class, feats, x, y_mask, y_class, sw_mask, sw_class, y_mask_perm, y_class_perm

    return losses, outs, perms

def trainIters(args):

    epoch_resume = 0
    model_dir = os.path.join('../models/', args.model_name)

    if args.resume:
        # will resume training the model with name args.model_name
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.model_name)

        epoch_resume = load_args.epoch_resume
        encoder = FeatureExtractor(load_args)
        decoder = RSIS(load_args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

        args = load_args

    elif args.transfer:
        # load model from args and replace last fc layer
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.transfer_from)
        encoder = FeatureExtractor(load_args)
        decoder = RSIS(args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

    else:
        encoder = FeatureExtractor(args)
        decoder = RSIS(args)

    # model checkpoints will be saved here
    make_dir(model_dir)

    # save parameters for future use
    pickle.dump(args, open(os.path.join(model_dir,'args.pkl'),'wb'))

    encoder_params = get_base_params(args,encoder)
    skip_params = get_skip_params(encoder)
    decoder_params = list(decoder.parameters()) + list(skip_params)
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params, args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params, args.weight_decay_cnn)

    if args.resume or args.transfer:
        enc_opt.load_state_dict(enc_opt_dict)
        dec_opt.load_state_dict(dec_opt_dict)
        from collections import defaultdict
        dec_opt.state = defaultdict(dict, dec_opt.state)

        # change fc layer for new classes
        if load_args.dataset != args.dataset and args.transfer:
            dim_in = decoder.fc_class.weight.size()[1]
            decoder.fc_class = nn.Linear(dim_in,args.num_classes)

    if not args.log_term:
        print "Training logs will be saved to:", os.path.join(model_dir, 'train.log')
        sys.stdout = open(os.path.join(model_dir, 'train.log'), 'w')
        sys.stderr = open(os.path.join(model_dir, 'train.err'), 'w')

    print args

    # objective functions for mask and class outputs.
    # these return the average across samples in batch whose value
    # needs to be considered (those where sw is 1)
    # mask_xentropy = BalancedStableMaskedBCELoss()
    mask_siou = softIoULoss()

    class_xentropy = MaskedNLLLoss(balance_weight=None)
    stop_xentropy = MaskedBCELoss(balance_weight=args.stop_balance_weight)

    if args.ngpus > 1 and args.use_gpu:
        decoder = torch.nn.DataParallel(decoder, device_ids=range(args.ngpus))
        encoder = torch.nn.DataParallel(encoder, device_ids=range(args.ngpus))
        mask_siou = torch.nn.DataParallel(mask_siou, device_ids=range(args.ngpus))
        class_xentropy = torch.nn.DataParallel(class_xentropy, device_ids=range(args.ngpus))
        stop_xentropy = torch.nn.DataParallel(stop_xentropy, device_ids=range(args.ngpus))
    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()
        class_xentropy.cuda()
        mask_siou.cuda()
        stop_xentropy.cuda()

    crits = [mask_siou, class_xentropy, stop_xentropy]
    optims = [enc_opt, dec_opt]

    torch.cuda.synchronize()
    start = time.time()

    # vars for early stopping
    best_val_loss = args.best_val_loss
    acc_patience = 0
    mt_val = -1

    # init windows to visualize, if visdom is enabled
    if args.visdom:
        import visdom
        viz = visdom.Visdom(port=args.port, server=args.server)
        lot, elot, mviz_pred, mviz_true, image_lot = init_visdom(args, viz)

    if args.curriculum_learning and epoch_resume == 0:
            args.limit_seqlen_to = 2

    # keep track of the number of batches in each epoch for continuity when plotting curves
    num_batches = {'train': 0, 'val': 0}
    for e in range(args.max_epoch):
        if e == 0 or args.multiscale:
            loaders, class_names = init_dataloaders(args)
        print "Epoch", e + epoch_resume
        # store losses in lists to display average since beginning
        epoch_losses = {'train': {'total': [], 'iou': [], 'stop': [], 'class': []},
                            'val': {'total': [], 'iou': [], 'stop': [], 'class': []}}
            # total mean for epoch will be saved here to display at the end
        total_losses = {'total': [], 'iou': [], 'stop': [], 'class': []}

        # check if it's time to do some changes here
        if e + epoch_resume >= args.finetune_after and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            args.update_encoder = True
            acc_patience = 0
            mt_val = -1
        if e + epoch_resume >= args.class_loss_after and not args.use_class_loss and not args.class_loss_after == -1:
            print("Starting to learn class loss")
            args.use_class_loss = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
            acc_patience = 0
            mt_val = -1
        if e + epoch_resume >= args.stop_loss_after and not args.use_stop_loss and not args.stop_loss_after == -1:
            if args.curriculum_learning:
                if args.limit_seqlen_to > args.min_steps:
                    print("Starting to learn stop loss")
                    args.use_stop_loss = True
                    best_val_loss = 1000 # reset because adding a loss term will increase the total value
                    acc_patience = 0
                    mt_val = -1
            else:
                print("Starting to learn stop loss")
                args.use_stop_loss = True
                best_val_loss = 1000 # reset because adding a loss term will increase the total value
                acc_patience = 0
                mt_val = -1

        # we validate after each epoch
        for split in ['train', 'val']:
            for batch_idx, (inputs, targets) in enumerate(loaders[split]):
                # send batch to GPU

                x, y_mask, y_class, sw_mask, sw_class = batch_to_var(args, inputs, targets)

                # we forward (and backward & update if training set)
                losses, outs, true_perm = runIter(args, encoder, decoder, x, y_mask,
                                                  y_class, sw_mask, sw_class,
                                                  crits, optims, mode=split)

                # store loss values in dictionary separately
                epoch_losses[split]['total'].append(losses[0])
                epoch_losses[split]['iou'].append(losses[1])
                epoch_losses[split]['stop'].append(losses[2])
                epoch_losses[split]['class'].append(losses[3])


                # print and display in visdom after some iterations
                if (batch_idx + 1)% args.print_every == 0:

                    mt = np.mean(epoch_losses[split]['total'])
                    mi = np.mean(epoch_losses[split]['iou'])
                    mc = np.mean(epoch_losses[split]['class'])
                    mx = np.mean(epoch_losses[split]['stop'])
                    if args.visdom:

                        if split == 'train':
                            # we display batch loss values in visdom (Training only)
                            viz.line(
                                X=torch.ones((1, 4)).cpu() * (batch_idx + e * num_batches[split]),
                                Y=torch.Tensor([mi, mx, mc, mt]).unsqueeze(0).cpu(),
                                win=lot,
                                update='append')
                        w = x.size()[-1]
                        h = x.size()[-2]
                        out_masks, out_classes, y_mask, y_class = outs_perms_to_cpu(args, outs, true_perm, h, w)

                        x = x.data.cpu().numpy()
                        # send image, sample predictions and ground truths to visdom
                        for t in range(np.shape(out_masks)[1]):
                            mask_pred = out_masks[0, t]
                            mask_true = y_mask[0, t]

                            class_pred = class_names[out_classes[0, t]]
                            class_true = class_names[y_class[0, t]]
                            mask_pred = np.reshape(mask_pred, (x.shape[-2], x.shape[-1]))
                            mask_true = np.reshape(mask_true, (x.shape[-2], x.shape[-1]))

                            # heatmap displays the mask upside down
                            viz.heatmap(np.flipud(mask_pred), win=mviz_pred[t],
                                        opts=dict(title='pred mask %d %s' % (t, class_pred)))
                            viz.heatmap(np.flipud(mask_true), win=mviz_true[t],
                                        opts=dict(title='true mask %d %s' % (t, class_true)))
                            viz.image((x[0] * 0.2 + 0.5) * 256, win=image_lot,
                                      opts=dict(title='image (unnnormalized)'))

                    te = time.time() - start
                    print "iter %d:\ttotal:%.4f\tclass:%.4f\tiou:%.4f\tstop:%.4f\ttime:%.4f" % (batch_idx, mt, mc, mi, mx, te)
                    torch.cuda.synchronize()
                    start = time.time()

            num_batches[split] = batch_idx + 1
            # compute mean val losses within epoch

            if split == 'val' and args.smooth_curves:
                if mt_val == -1:
                    mt = np.mean(epoch_losses[split]['total'])
                else:
                    mt = 0.9*mt_val + 0.1*np.mean(epoch_losses[split]['total'])
                mt_val = mt

            else:
                mt = np.mean(epoch_losses[split]['total'])
            
            mi = np.mean(epoch_losses[split]['iou'])
            mc = np.mean(epoch_losses[split]['class'])
            mx = np.mean(epoch_losses[split]['stop'])


            # save train and val losses for the epoch to display in visdom
            total_losses['iou'].append(mi)
            total_losses['class'].append(mc)
            total_losses['stop'].append(mx)
            total_losses['total'].append(mt)

            args.epoch_resume = e + epoch_resume

            print "Epoch %d:\ttotal:%.4f\tclass:%.4f\tiou:%.4f\tstop:%.4f\t(%s)" % (e, mt, mc, mi,mx, split)

        # epoch losses
        if args.visdom:
            update = True if e == 0 else 'append'
            for l in ['total', 'iou', 'stop', 'class']:
                viz.line(X=torch.ones((1, 2)).cpu() * (e + 1),
                         Y=torch.Tensor(total_losses[l]).unsqueeze(0).cpu(),
                         win=elot[l],
                         update=update)

        if mt < (best_val_loss - args.min_delta):
            print "Saving checkpoint."
            best_val_loss = mt
            args.best_val_loss = best_val_loss
            # saves model, params, and optimizers
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt)
            acc_patience = 0
        else:
            acc_patience += 1

        if acc_patience > args.patience and not args.use_class_loss and not args.class_loss_after == -1:
            print("Starting to learn class loss")
            acc_patience = 0
            args.use_class_loss = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
            mt_val = -1
            encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, _ = load_checkpoint(args.model_name)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            enc_opt.load_state_dict(enc_opt_dict)
            dec_opt.load_state_dict(dec_opt_dict)
        if acc_patience > args.patience and args.curriculum_learning and args.limit_seqlen_to < args.maxseqlen:
            print("Adding one step more:")
            acc_patience = 0
            args.limit_seqlen_to += args.steps_cl
            print(args.limit_seqlen_to)
            best_val_loss = 1000
            mt_val = -1

        if acc_patience > args.patience and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            acc_patience = 0
            args.update_encoder = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
            mt_val = -1
            encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, _ = load_checkpoint(args.model_name)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            enc_opt.load_state_dict(enc_opt_dict)
            dec_opt.load_state_dict(dec_opt_dict)
        if acc_patience > args.patience and not args.use_stop_loss and not args.stop_loss_after == -1:
            if args.curriculum_learning:
                print("Starting to learn stop loss")
                if args.limit_seqlen_to > args.min_steps:
                    acc_patience = 0
                    args.use_stop_loss = True
                    best_val_loss = 1000 # reset because adding a loss term will increase the total value
                    mt_val = -1
            else:
                print("Starting to learn stop loss")
                acc_patience = 0
                args.use_stop_loss = True
                best_val_loss = 1000 # reset because adding a loss term will increase the total value
                mt_val = -1

            encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, _ = load_checkpoint(args.model_name)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            enc_opt.load_state_dict(enc_opt_dict)
            dec_opt.load_state_dict(dec_opt_dict)
        # early stopping after N epochs without improvement
        if acc_patience > args.patience_stop:
            break


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    #np.random.seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    trainIters(args)
