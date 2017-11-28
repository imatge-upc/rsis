from args import get_parser
from utils.hungarian import match, softIoU
from utils.utils import get_optimizer, batch_to_var, make_dir, load_checkpoint
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as data
import os
import warnings
import math
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def test(args, encoder, decoder, x, y_mask, y_class, sw_mask, sw_class):

    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """

    T = args.maxseqlen
    hidden = None
    loss_mask_iou = 0
    loss_mask_x = 0
    loss_class = 0

    out_masks = []
    out_masks_flat = []
    out_classes = []
    out_stops = []
    encoder.eval()
    decoder.eval()

    feats = encoder(x)
    scores = torch.zeros(y_mask.size(0),args.gt_maxseqlen,args.maxseqlen)
    # loop over sequence length and get predictions
    for t in range(0, T):
        out_mask, out_class, out_stop, hidden = decoder(feats, hidden)
        upsample_match = torch.nn.UpsamplingBilinear2d(size = (x.size()[-2],x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask_flat = out_mask.view(out_mask.size(0), -1)

        # repeat predicted mask as many times as elements in ground truth.
        # to compute iou against all ground truth elements at once
        # WARNING: masks will be matched based on softIoU only ! Even though
        # we use BCE loss for training.
        y_pred_i = out_mask_flat.unsqueeze(0)
        y_pred_i = y_pred_i.permute(1,0,2)
        y_pred_i = y_pred_i.repeat(1,y_mask.size(1),1)
        y_pred_i = y_pred_i.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))
        y_true_p = y_mask.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))
        sw_mask_p = sw_mask.view(sw_mask.size(0)*sw_mask.size(1)).float()

        c = softIoU(y_true_p, y_pred_i)
        c = c.view(sw_mask.size(0),-1)
        scores[:,:,t] = c.cpu().data

        # get predictions in list to concat later
        out_masks.append(out_mask)
        out_masks_flat.append(out_mask_flat)
        out_classes.append(out_class)
        out_stops.append(out_stop)
    # concat all outputs into single tensor to compute the loss
    out_masks = torch.cat(out_masks,1)
    out_classes = torch.cat(out_classes,1).view(out_class.size(0),len(out_classes),-1)
    out_stops = torch.cat(out_stops,1).view(out_stop.size(0),len(out_stops),-1)
    out_masks_flat = torch.cat(out_masks_flat,1).view(out_mask_flat.size(0),len(out_masks_flat),-1)

    # get permutations of ground truth based on predictions (CPU computation)
    masks = [y_mask,out_masks_flat]
    classes = [y_class,out_classes]
    # mask predictions in positions > length of sequence so that they are not picked
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
    y_mask_perm = Variable(torch.from_numpy(y_mask_perm),requires_grad=False)
    y_class_perm = Variable(torch.from_numpy(y_class_perm),requires_grad=False)
    if args.use_gpu:
        y_mask_perm = y_mask_perm.cuda()
        y_class_perm = y_class_perm.cuda()

    outs = [torch.sigmoid(out_masks).data, out_classes.data]
    return outs, [y_mask_perm.data,y_class_perm.data], torch.sigmoid(out_stops).data
