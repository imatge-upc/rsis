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
from torch.nn.modules.upsampling import UpsamplingBilinear2d
sys.path.append("..")
from utils.utils import get_skip_dims


class FeatureExtractor(nn.Module):
    #Returns base network to extract visual features from image

    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        '''
        self.base = Trunk('', '')
        '''
        self.base = ResNet50()
        self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())

        self.conv_embed = nn.ModuleList()
        dims = [2048, 1024, 512, 256]
        for dim in dims:
            hidden_dim = args.hidden_size//len(dims)
            conv = nn.Sequential(nn.Dropout2d(args.dropout),
                                 nn.Conv2d(dim, hidden_dim, 1, padding=0))
            self.conv_embed.append(conv)

        self.pyramid_poolings = nn.Sequential(
            nn.Conv2d(args.hidden_size, args.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(args.hidden_size))

    def forward(self, x, keep_grad=True):
        if keep_grad:
            feats = self.base(x)
        else:
            with torch.no_grad():
                feats = self.base(x)

        embed_feats = []
        up_ = UpsamplingBilinear2d(size=(x.size(-2) // 8, x.size(-1) // 8))
        for i, conv in enumerate(self.conv_embed):
            conv(feats[i])
            embed_feats.append(up_(conv(feats[i])))
        embed_feats = torch.cat(embed_feats, 1)
        return self.pyramid_poolings(embed_feats)


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
            clstm_i = ConvLSTMCell(args, self.hidden_size, self.hidden_size, self.kernel_size, padding=padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(self.hidden_size, 1, self.kernel_size, padding=padding)
        self.box_out = nn.Conv2d(self.hidden_size, 2, self.kernel_size, padding=padding)
        self.fc_class = nn.Linear(self.hidden_size, self.num_classes)
        self.fc_stop = nn.Linear(self.hidden_size, 1)
        self.class_conv_merge = nn.Sequential(nn.Conv2d(self.hidden_size*2, self.hidden_size, self.kernel_size,
                                                        padding=padding))
        self.dp_if_train = nn.Dropout2d(self.dropout)

    def forward(self, feats, prev_hidden_list):

        hidden_list = []
        feats = self.dp_if_train(feats)

        lstm_feats = feats
        for i in range(len(self.clstm_list)):

            # hidden states will be initialized the first time forward is called
            if prev_hidden_list is None:
                state = self.clstm_list[i](lstm_feats, None)
            else:
                # else we take the ones from the previous step for the forward pass
                state = self.clstm_list[i](lstm_feats, prev_hidden_list[i])
            hidden_list.append(state)
            hidden = state[0]

            lstm_feats = hidden
            lstm_feats = self.dp_if_train(lstm_feats)

        out_mask = self.conv_out(lstm_feats)
        out_box = self.box_out(lstm_feats)

        class_feats = torch.cat([lstm_feats, feats], 1)
        class_feats = self.class_conv_merge(class_feats)
        fc_feats_cls = nn.MaxPool2d(class_feats.size()[2:])(class_feats).view(class_feats.size(0), class_feats.size(1))
        fc_feats = nn.MaxPool2d(lstm_feats.size()[2:])(lstm_feats).view(lstm_feats.size(0), lstm_feats.size(1))
        # classification branch
        # fc_feats = torch.nn.functional.dropout(fc_feats, p=self.dropout_cls, training=self.training)
        class_feats = nn.Softmax(dim=-1)(self.fc_class(fc_feats_cls))
        stop_probs = self.fc_stop(fc_feats)

        return out_mask, out_box, class_feats, stop_probs, hidden_list


