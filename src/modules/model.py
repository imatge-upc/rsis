import torch
import torch.nn as nn
from .clstm import ConvLSTMCell
import argparse
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import math
from .vision import VGG16, ResNet34, ResNet50, ResNet101
import sys
from torch.nn.modules.upsampling import UpsamplingBilinear2d
sys.path.append("..")
from utils.utils import get_skip_dims


class FeatureExtractor(nn.Module):
    #Returns base network to extract visual features from image

    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        if args.base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        else:
            self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())

        self.conv_embed = nn.ModuleList()
        padding = 0 if args.kernel_size == 1 else 1
        dims = [2048, 1024, 512, 256, 64]
        out_dims = [args.hidden_size, args.hidden_size, args.hidden_size//2, args.hidden_size//4, args.hidden_size//8]
        for i, dim in enumerate(dims):
            conv = nn.Sequential(nn.Conv2d(dim, out_dims[i], args.kernel_size, padding=padding),
                                 nn.BatchNorm2d(out_dims[i]))
            self.conv_embed.append(conv)

    def forward(self, x, keep_grad=True):
        if keep_grad:
            feats = self.base(x)
        else:
            with torch.no_grad():
                feats = self.base(x)

        embed_feats = []
        # up_ = UpsamplingBilinear2d(size=(x.size(-2) // 8, x.size(-1) // 8))
        for i, conv in enumerate(self.conv_embed):
            conv(feats[i])
            embed_feats.append(conv(feats[i]))
        # embed_feats = torch.cat(embed_feats, 1)
        return embed_feats


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
        self.skip_mode = args.skip_mode

        skip_dims_out = [self.hidden_size, self.hidden_size // 2,
                         self.hidden_size // 4, self.hidden_size // 8,
                         self.hidden_size // 16]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 5 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i - 1]
                clstm_in_dim *= 2

            clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i], self.kernel_size, padding=padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(skip_dims_out[-1], 1, self.kernel_size, padding=padding)
        self.conv_box = nn.Conv2d(skip_dims_out[-1], 2, self.kernel_size, padding=padding)
        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim += sk

        self.fc_stop = nn.Sequential(nn.Linear(fc_dim, self.hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_size, self.hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_size, 1)
                                     )
        self.fc_class = nn.Linear(fc_dim, self.num_classes)

        self.dp_if_train = nn.Dropout2d(self.dropout)
        self.dropout_cls = args.dropout_cls

    def forward(self, feats, prev_hidden_list):

        clstm_in = feats[0]
        feats = feats[1:]
        side_feats = []
        hidden_list = []

        # feats_conv5 = clstm_in
        # feats_conv5_pool = nn.MaxPool2d(clstm_in.size()[2:])(feats_conv5)

        for i in range(len(feats) + 1):

            # hidden states will be initialized the first time forward is called
            if prev_hidden_list is None:
                state = self.clstm_list[i](clstm_in, None)
            else:
                # else we take the ones from the previous step for the forward pass
                state = self.clstm_list[i](clstm_in, prev_hidden_list[i])
            hidden_list.append(state)
            hidden = state[0]
            hidden = self.dp_if_train(hidden)

            side_feats.append(nn.MaxPool2d(clstm_in.size()[2:])(hidden))

            # apply skip connection
            if i < len(feats):
                skip_vec = feats[i]
                upsample = nn.UpsamplingBilinear2d(size=(skip_vec.size()[-2], skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                clstm_in = torch.cat([hidden, skip_vec], 1)
            else:
                up = nn.UpsamplingBilinear2d(size=(hidden.size()[-2] * 2, hidden.size()[-1] * 2))
                hidden = up(hidden)
                clstm_in = hidden

        out_mask = self.conv_out(clstm_in)
        out_box = self.conv_box(nn.MaxPool2d(8)(clstm_in))

        # classification branch
        side_feats = torch.cat(side_feats, 1).squeeze()
        stop_probs = self.fc_stop(side_feats.detach())
        side_feats = torch.nn.functional.dropout(side_feats, p=self.dropout_cls, training=self.training)

        class_feats = self.fc_class(side_feats)

        # the log is computed in the objective function
        class_probs = nn.Softmax(dim=-1)(class_feats)

        return out_mask, out_box, class_probs, stop_probs, hidden_list


