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
from modules.coordconv import CoordConv
import sys
from torch.nn.modules.upsampling import UpsamplingBilinear2d
sys.path.append("..")
from utils.utils import get_skip_dims


class FeatureExtractor(nn.Module):

    # Returns base network to extract visual features from image

    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        if args.base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        else:
            self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())

        conv_operator = CoordConv if args.coordconv else nn.Conv2d

        self.conv_embed = nn.ModuleList()
        dims = [2048, 1024, 512, 256, 64]
        # out_dims = [args.hidden_size, args.hidden_size, args.hidden_size//2, args.hidden_size//4, args.hidden_size//8]
        for i, dim in enumerate(dims):
            if args.coordconv:
                dim = dim+2
            conv = nn.Sequential(conv_operator(in_channels=dim, out_channels=args.hidden_size,
                                               kernel_size=1, padding=0, bias=False),
                                 nn.BatchNorm2d(args.hidden_size),
                                 nn.ReLU())
            self.conv_embed.append(conv)

    def forward(self, x, keep_grad=True):
        if keep_grad:
            feats = self.base(x)
        else:
            with torch.no_grad():
                feats = self.base(x)

        embed_feats = []
        up_ = UpsamplingBilinear2d(size=(x.size(-2) // 8, x.size(-1) // 8))
        for i, conv in enumerate(self.conv_embed):
            feat = conv(feats[i])
            if feat.size(-2) > x.size(-2) // 8:
                feat = up_(feat)
            embed_feats.append(feat)
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

        self.skip_mode = args.skip_mode

        skip_dims_out = [self.hidden_size, self.hidden_size // 2,
                         self.hidden_size // 4, self.hidden_size // 8,
                         self.hidden_size // 16]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 5 is the number of deconv steps that we need to reach image size in the output
        for i in range(5):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = self.hidden_size
                clstm_in_dim *= 2

            clstm_i = ConvLSTMCell(args, clstm_in_dim + 1, self.hidden_size, self.kernel_size, padding=padding)
            self.clstm_list.append(clstm_i)

        value = 10
        self.bias = nn.Parameter(torch.ones(1)*value)

        self.conv_out = nn.Conv2d(self.hidden_size, 1, self.kernel_size, padding=padding)
        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 5*self.hidden_size

        self.fc_stop = nn.Sequential(nn.Linear(fc_dim, self.hidden_size),
                                     nn.BatchNorm1d(self.hidden_size, momentum=0.01),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_size, 1)
                                     )
        self.fc_class = nn.Linear(fc_dim, self.num_classes)

    def forward(self, feats, prev_hidden_list, prev_masks):

        clstm_in = feats[0]
        feats = feats[1:]
        side_feats = []
        hidden_list = []

        for i in range(len(feats) + 1):
            upsample_match = nn.UpsamplingBilinear2d(size=(clstm_in.size()[-2], clstm_in.size()[-1]))
            prev_mask_i = upsample_match(prev_masks)

            clstm_in = torch.cat([clstm_in, prev_mask_i], 1)

            # hidden states will be initialized the first time forward is called
            if prev_hidden_list is None:
                state = self.clstm_list[i](clstm_in, None)
            else:
                # else we take the ones from the previous step for the forward pass
                state = self.clstm_list[i](clstm_in, prev_hidden_list[i])
            hidden_list.append(state)
            hidden = state[0]

            side_feats.append(nn.MaxPool2d(clstm_in.size()[2:])(hidden))

            # apply skip connection
            if i < len(feats):
                skip_vec = feats[i]
                up_ = nn.UpsamplingBilinear2d(size=(skip_vec.size()[-2], skip_vec.size()[-1]))
                hidden = up_(hidden)
                # skip connection
                clstm_in = torch.cat([hidden, skip_vec], 1)
            else:
                clstm_in = hidden

        out_mask = self.conv_out(clstm_in)
        bs, c, h, w = out_mask.size()

        out_mask = nn.functional.log_softmax(out_mask.view(out_mask.size(0), -1), dim=-1)
        out_mask = out_mask.view(bs, c, h, w) + self.bias

        # classification branch
        side_feats = torch.cat(side_feats, 1).squeeze()
        stop_probs = self.fc_stop(side_feats.detach())
        class_feats = self.fc_class(side_feats)

        # the log is computed in the objective function
        class_probs = nn.Softmax(dim=-1)(class_feats)

        return out_mask, class_probs, stop_probs, hidden_list
