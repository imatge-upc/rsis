import torch
import torch.nn as nn
from clstm import ConvLSTMCell
import argparse
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import math
from vision import VGG16, ResNet34, ResNet50, ResNet101
import sys
sys.path.append("..")
from utils.utils import get_skip_dims

class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self,args):
        super(FeatureExtractor,self).__init__()
        skip_dims_in = get_skip_dims(args.base_model)

        if args.base_model == 'resnet34':
            self.base = ResNet34()
            self.base.load_state_dict(models.resnet34(pretrained=True).state_dict())
        elif args.base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        elif args.base_model == 'resnet101':
            self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())
        elif args.base_model == 'vgg16':
            self.base = VGG16()
            self.base.load_state_dict(models.vgg16(pretrained=True).state_dict())

        else:
            raise Exception("The base model you chose is not supported !")

        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.padding = 0 if self.kernel_size == 1 else 1

        self.sk5 = nn.Conv2d(skip_dims_in[0],self.hidden_size,self.kernel_size,padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1],self.hidden_size,self.kernel_size,padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2],self.hidden_size/2,self.kernel_size,padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3],self.hidden_size/4,self.kernel_size,padding=self.padding)
        self.sk1 = nn.Conv2d(skip_dims_in[4],self.hidden_size/8,self.kernel_size,padding=self.padding)


        self.bn5 = nn.BatchNorm2d(self.hidden_size)
        self.bn4 = nn.BatchNorm2d(self.hidden_size)
        self.bn3 = nn.BatchNorm2d(self.hidden_size/2)
        self.bn2 = nn.BatchNorm2d(self.hidden_size/4)
        self.bn1 = nn.BatchNorm2d(self.hidden_size/8)

    def forward(self,x,semseg=False, raw = False):
        x5,x4,x3,x2,x1 = self.base(x)

        x5_skip = self.bn5(self.sk5(x5))
        x4_skip = self.bn4(self.sk4(x4))
        x3_skip = self.bn3(self.sk3(x3))
        x2_skip = self.bn2(self.sk2(x2))
        x1_skip = self.bn1(self.sk1(x1))

        if semseg:
            return x5
        elif raw:
            return x5, x4, x3, x2, x1
        else:
            return x5_skip, x4_skip, x3_skip, x2_skip, x1_skip

class RSIS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):
        super(RSIS,self).__init__()

        if hasattr(args, 'rnn_type'):
            self.rnn_type = args.rnn_type
        else:
            self.rnn_type = 'lstm'

        if hasattr(args, 'conv_start'):
            self.conv_start = args.conv_start
        else:
            self.conv_start = 0
        skip_dims_in = get_skip_dims(args.base_model)
        self.input_dim = skip_dims_in[0]
        self.input_size = args.imsize
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.kernel_size = args.kernel_size
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = args.dropout
        self.dropout_stop = args.dropout_stop
        self.dropout_cls = args.dropout_cls
        self.batchnorm = args.batchnorm
        self.skip_mode = args.skip_mode

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, self.hidden_size/2,
                         self.hidden_size/4,self.hidden_size/8,
                         self.hidden_size/16]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 5 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2

            clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)

        #self.last_bn = nn.BatchNorm2d(skip_dims_out[-1])
        self.conv_out = nn.Conv2d(skip_dims_out[-1], 1,self.kernel_size, padding = padding)

        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk

        #conv_stop_in = fc_dim + 1 if args.use_feedback else fc_dim
        self.fc_class = nn.Linear(fc_dim,self.num_classes)
        self.fc_stop = nn.Linear(fc_dim,1)
        # classification branch
        #self.conv_class = nn.Conv2d(fc_dim,self.num_classes,self.kernel_size,padding=padding)
        #self.conv_stop = nn.Conv2d(conv_stop_in,1,1,padding=0)

    def forward(self, skip_feats, prev_hidden_list):

        clstm_in = skip_feats[0]
        skip_feats = skip_feats[1:]
        side_feats = []
        hidden_list = []

        for i in range(len(skip_feats)+1):

            # hidden states will be initialized the first time forward is called
            if prev_hidden_list is None:
                state = self.clstm_list[i](clstm_in,None)
            else:
                # else we take the ones from the previous step for the forward pass
                state = self.clstm_list[i](clstm_in,prev_hidden_list[i])
            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            side_feats.append(nn.MaxPool2d(clstm_in.size()[2:])(hidden))
            #pool_side_feats = nn.AdaptiveMaxPool2d(output_size=skip_feats[0].size()[2:])
            #side_feats.append(pool_side_feats(hidden))

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2],skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden,skip_vec],1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden*skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                hidden = self.upsample(hidden)
                clstm_in = hidden

        #clstm_in = self.last_bn(clstm_in)
        out_mask = self.conv_out(clstm_in)
        # classification branch
        side_feats = torch.cat(side_feats,1).squeeze()
        if self.dropout_cls > 0:
            class_feats = nn.Dropout(self.dropout_cls)(side_feats)
        else:
            class_feats = side_feats
        class_feats = self.fc_class(class_feats)
        if self.dropout_stop > 0:
            stop_feats = nn.Dropout(self.dropout_stop)(side_feats)
        else:
            stop_feats = side_feats
        stop_probs = self.fc_stop(stop_feats)
        #class_feats = nn.AdaptiveMaxPool2d(1)(class_feats).squeeze()
        #stop_probs = nn.AdaptiveMaxPool2d(1)(stop_probs).squeeze().view(-1,1)

        # the log is computed in the objective function
        class_probs = nn.Softmax()(class_feats)

        return out_mask, class_probs, stop_probs, hidden_list
