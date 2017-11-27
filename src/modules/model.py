import torch
import torch.nn as nn
from clstm import ConvLSTMCell, ConvGRUCell
import argparse
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import math
from vision import ResNet50, ResNet34, VGG16, ResNet101, WideResNet34, WideResNet50, WideResNet101
import sys
sys.path.append("..")
from utils.utils import get_skip_dims
import drn
from segment import DRNSeg
from deeplab import Classifier_Module

class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self,args):
        super(FeatureExtractor,self).__init__()
        skip_dims_in = get_skip_dims(args.base_model)

        if args.base_model == 'resnet34':
            if args.wideresnet:
                self.base = WideResNet34()
            else:
                self.base = ResNet34()
            self.base.load_state_dict(models.resnet34(pretrained=True).state_dict())
        elif args.base_model == 'resnet50':
            if args.wideresnet:
                self.base = WideResNet50()
            else:
                self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        elif args.base_model == 'resnet101':
            if args.wideresnet:
                self.base = WideResNet101()
            else:
                self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())
        elif args.base_model == 'vgg16':
            self.base = VGG16()
            self.base.load_state_dict(models.vgg16(pretrained=True).state_dict())
        elif 'drn' in args.base_model:
            self.base = DRNSeg(args.base_model, classes=19,
                            pretrained_model=None, pretrained=False,
                            use_torch_up=False)
            if args.drn_seg:
                pretrained_dict = torch.load(args.seg_checkpoint_path)
                model_dict = self.base.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                self.base.load_state_dict(pretrained_dict)

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

class RIASS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):
        super(RIASS,self).__init__()

        if hasattr(args, 'rnn_type'):
            self.rnn_type = args.rnn_type
        else:
            self.rnn_type = 'lstm'

        if hasattr(args, 'conv_start'):
            self.conv_start = args.conv_start
        else:
            self.conv_start = 0
        self.nconvlstm = args.nconvlstm
        skip_dims_in = get_skip_dims(args.base_model)
        skip_dims_in = skip_dims_in[self.conv_start:self.conv_start+self.nconvlstm]
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
        self.D = args.D
        self.limit_width = args.limit_width

        if hasattr(args, 'rnn_type'):
            self.rnn_type = args.rnn_type
        else:
            self.rnn_type = 'lstm'

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, self.hidden_size/2,
                         self.hidden_size/4,self.hidden_size/8,
                         self.hidden_size/16][0:self.nconvlstm]

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 5 is the number of deconv steps that we need to reach image size in the output
        for i in range(self.nconvlstm):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2

            if args.use_feedback:
                clstm_in_dim = clstm_in_dim + 1 # +1 for additional channel containing canvas of masks

            if self.rnn_type == 'lstm':
                clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            else:
                clstm_i = ConvGRUCell(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
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

    def forward(self, skip_feats, prev_hidden_list, prev_mask = None, return_gates = False):

        skip_feats = skip_feats[self.conv_start:self.conv_start+self.nconvlstm]
        clstm_in = skip_feats[0]
        skip_feats = skip_feats[1:]
        side_feats = []
        hidden_list = []

        if return_gates:
            gate_list = []

        for i in range(len(skip_feats)+1):

            if prev_mask is not None:
                # add the previous mask as a channel in the conv lstm's input
                downsample = nn.AdaptiveMaxPool2d(output_size=clstm_in.size()[2:])
                prev_mask_i = downsample(prev_mask)
                # concat prev_mask
                clstm_in = torch.cat([clstm_in,prev_mask_i],1)

            # hidden states will be initialized the first time forward is called
            if prev_hidden_list is None:
                if return_gates:
                    state,in_gate,remember_gate = self.clstm_list[i](clstm_in,None,return_gates)
                    gate_list.append([in_gate,remember_gate])
                else:
                    state = self.clstm_list[i](clstm_in,None,return_gates)
            else:
                # else we take the ones from the previous step for the forward pass
                if return_gates:
                    state,in_gate,remember_gate = self.clstm_list[i](clstm_in,prev_hidden_list[i],return_gates)
                    gate_list.append([in_gate,remember_gate])
                else:
                    state = self.clstm_list[i](clstm_in,prev_hidden_list[i],return_gates)
            hidden_list.append(state)

            if self.rnn_type == 'lstm':
                hidden = state[0]
            else:
                hidden = state

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            side_feats.append(nn.MaxPool2d(clstm_in.size()[2:])(hidden))
            #pool_side_feats = nn.AdaptiveMaxPool2d(output_size=skip_feats[0].size()[2:])
            #side_feats.append(pool_side_feats(hidden))

            # apply skip connection
            if i < len(skip_feats):
                if self.limit_width and i>1:
                    resize_skip = nn.UpsamplingBilinear2d((hidden.size()[-2],hidden.size()[-1]))
                    skip_vec = resize_skip(skip_feats[i])
                else:
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
                if not self.limit_width:
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

        if return_gates:
            return out_mask, class_probs, stop_probs, hidden_list, gate_list
        else:
            return out_mask, class_probs, stop_probs, hidden_list

class SemanticSegmentation(nn.Module):

    def __init__(self,args):
        super(SemanticSegmentation, self).__init__()
        skip_dims_in = get_skip_dims(args.base_model)
        self.conv1 = nn.Conv2d(skip_dims_in[0], args.num_classes, 1, padding = 0,bias=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        self.softmax = nn.LogSoftmax()

    def forward(self,feat):

        return self.softmax(self.up(self.conv1(feat)))
