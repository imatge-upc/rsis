from args import get_parser
from utils.hungarian import match, softIoU
from utils.utils import batch_to_var, make_dir, load_checkpoint
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


def test(args, encoder, decoder, x):

    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """

    T = args.maxseqlen
    hidden = None
    out_masks, out_classes, out_stops, out_boxes = [], [], [], []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        feats = encoder(x)
    # loop over sequence length and get predictions
    feed_masks = torch.zeros(x.size(0), x.size()[-2] * x.size()[-1]).cuda()
    for t in range(0, T):
        out_mask, out_box, out_class, out_stop, hidden = decoder(feats, hidden,
                                                                 feed_masks.view(x.size(0), 1, x.size()[-2], x.size()[-1]))
        upsample_match = torch.nn.UpsamplingBilinear2d(size = (x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        # get predictions in list to concat later
        out_masks.append(out_mask)
        out_classes.append(out_class)
        out_stops.append(out_stop)
        out_boxes.append(out_box.unsqueeze(1))

        if args.use_feedback:
            feed_masks = (torch.sigmoid(out_mask) > 0.5).float().detach()

    # concat all outputs into single tensor to compute the loss
    out_masks = torch.cat(out_masks,1)
    out_classes = torch.cat(out_classes,1).view(out_class.size(0), len(out_classes), -1)
    out_stops = torch.cat(out_stops,1).view(out_stop.size(0), len(out_stops), -1)
    out_boxes = torch.cat(out_boxes, 1)

    return torch.sigmoid(out_masks).data, out_boxes.data, out_classes.data, torch.sigmoid(out_stops).data
