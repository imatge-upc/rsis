import torch
import torch.nn as nn
from .clstm import ConvLSTMCell
import argparse
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import math
from modules.pspnet.models.models import Trunk
from .vision import VGG16, ResNet34, ResNet50, ResNet101
import sys
sys.path.append("..")
from utils.utils import get_skip_dims


class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()

        self.base = Trunk('', '')

        self.pyramid_poolings = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, args.hidden_size, kernel_size=1))

    def forward(self, x):

        x = self.base(x)
        return self.pyramid_poolings(x)


class RNNDecoder(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):
        super(RNNDecoder, self).__init__()

        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.kernel_size = args.kernel_size
        self.num_lstms = args.num_lstms
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = args.dropout
        self.dropout_stop = args.dropout_stop
        self.dropout_cls = args.dropout_cls
        self.skip_mode = args.skip_mode

        self.clstm_list = nn.ModuleList()
        for i in range(self.num_lstms):
            clstm_i = ConvLSTMCell(args, self.hidden_size, self.hidden_size,self.kernel_size, padding=padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(self.hidden_size, 1, self.kernel_size, padding=padding)
        self.box_out = nn.Conv2d(self.hidden_size, 2, self.kernel_size, padding=padding)
        self.fc_class = nn.Linear(self.hidden_size, self.num_classes)
        self.fc_stop = nn.Linear(self.hidden_size, 1)

    def forward(self, feats, prev_hidden_list):

        hidden_list = []

        for i in range(len(self.clstm_list)):

            # hidden states will be initialized the first time forward is called
            if prev_hidden_list is None:
                state = self.clstm_list[i](feats, None)
            else:
                # else we take the ones from the previous step for the forward pass
                state = self.clstm_list[i](feats, prev_hidden_list[i])
            hidden_list.append(state)
            hidden = state[0]

            feats = hidden

        out_mask = self.conv_out(feats)
        out_box = self.box_out(feats)

        fc_feats = nn.MaxPool2d(feats.size()[2:])(feats).view(feats.size(0), feats.size(1))
        # classification branch
        class_feats = self.fc_class(fc_feats)
        stop_probs = self.fc_stop(fc_feats)

        return out_mask, out_box, class_feats, stop_probs, hidden_list


