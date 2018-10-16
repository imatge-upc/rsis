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
    out_masks, out_classes, out_stops, out_boxes, uncerts = [], [], [], [], []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        feats = encoder(x)
    # loop over sequence length and get predictions
    for t in range(0, T):
        out_mask, out_box, out_class, out_stop, uncertainty, hidden = decoder(feats, hidden)
        upsample_match = torch.nn.UpsamplingBilinear2d(size = (x.size()[-2],x.size()[-1]))
        out_mask = upsample_match(out_mask)
        # get predictions in list to concat later
        out_masks.append(out_mask)
        out_classes.append(out_class)
        out_stops.append(out_stop)
        out_boxes.append(out_box.unsqueeze(1))
        uncerts.append(uncertainty)
    # concat all outputs into single tensor to compute the loss
    out_masks = torch.cat(out_masks,1)
    out_classes = torch.cat(out_classes,1).view(out_class.size(0), len(out_classes), -1)
    out_stops = torch.cat(out_stops,1).view(out_stop.size(0), len(out_stops), -1)
    uncerts = torch.cat(uncerts, 1).view(uncertainty.size(0), len(uncerts), -1)
    out_boxes = torch.cat(out_boxes, 1)

    return torch.sigmoid(out_masks).data, out_boxes.data, out_classes.data, torch.sigmoid(out_stops).data, torch.sigmoid(uncerts).data
