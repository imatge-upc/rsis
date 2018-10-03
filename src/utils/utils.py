from torch.autograd import Variable
import torch
import os
import numpy as np
import pickle
from collections import OrderedDict

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def check_parallel(encoder_dict,decoder_dict):
	# check if the model was trained using multiple gpus
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break;
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict

def save_checkpoint(args, encoder, decoder, optimizer):
    torch.save(encoder.state_dict(), os.path.join('../models',args.model_name,'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../models',args.model_name,'decoder.pt'))
    torch.save(optimizer.state_dict(), os.path.join('../models',args.model_name,'optimizer.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../models',args.model_name,'args.pkl'),'wb'))

def load_checkpoint(model_name,use_gpu=True):
    if use_gpu:
        encoder_dict = torch.load(os.path.join('../models',model_name,'encoder.pt'))
        decoder_dict = torch.load(os.path.join('../models',model_name,'decoder.pt'))
        optimizer_dict = torch.load(os.path.join('../models',model_name,'optimizer.pt'))
    else:
        encoder_dict = torch.load(os.path.join('../models',model_name,'encoder.pt'), map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join('../models',model_name,'decoder.pt'), map_location=lambda storage, location: storage)
        optimizer_dict = torch.load(os.path.join('../models',model_name,'optimizer.pt'), map_location=lambda storage, location: storage)
    # save parameters for future use
    args = pickle.load(open(os.path.join('../models',model_name,'args.pkl'),'rb'))

    return encoder_dict, decoder_dict, optimizer_dict, args

def batch_to_var(args, inputs, targets):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs,requires_grad=False)
    y_mask = Variable(targets[:,:,:-3].float(),requires_grad=False)
    y_class = Variable(targets[:,:,-3].long(),requires_grad=False)
    sw_mask = Variable(targets[:,:,-2],requires_grad=False)
    sw_class = Variable(targets[:,:,-1],requires_grad=False)

    if args.use_gpu:
        return x.cuda(), y_mask.cuda(), y_class.cuda(), sw_mask.cuda(), sw_class.cuda()
    else:
        return x, y_mask, y_class, sw_mask, sw_class

def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101':
        skip_dims_in = [2048,1024,512,256,64]
    elif model_name == 'resnet34':
        skip_dims_in = [512,256,128,64,64]
    elif model_name =='vgg16':
        skip_dims_in = [512,512,256,128,64]

    return skip_dims_in

def init_visdom(args,viz):

    # initialize visdom figures

    lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,4)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Training Losses',
            legend=['iou','xentr','class','total']
        )
    )

    elot = {}
    # epoch losses
    elot['class'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Class Loss',
            legend = ['train','val']
        )
    )

    elot['iou'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='sIoU Loss',
            legend = ['train','val']
        )
    )

    elot['stop'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Stop Loss',
            legend = ['train','val']
        )
    )

    elot['total'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Total Loss',
            legend = ['train','val']
        )
    )

    mviz_pred = {}
    for i in range(args.maxseqlen):
        mviz_pred[i] = viz.heatmap(X=np.zeros((args.imsize,args.imsize)),
                                   opts=dict(title='Pred mask t'))

    mviz_true = {}
    for i in range(args.maxseqlen):
        mviz_true[i] = viz.heatmap(X=np.zeros((args.imsize,args.imsize)),
                                   opts=dict(title='True mask t'))


    image_lot = viz.image(np.ones((3,args.imsize,args.imsize)),
                        opts=dict(title='image'))


    return lot, elot, mviz_pred, mviz_true, image_lot

def outs_perms_to_cpu(args,outs,true_perm,h,w):
    # ugly function that turns contents of torch variables to numpy
    # (used for display during training)

    out_masks = outs[0]
    y_mask_perm = true_perm[0]

    y_mask_perm = y_mask_perm.view(y_mask_perm.size(0),y_mask_perm.size(1),h,w)
    out_masks = out_masks.view(out_masks.size(0),out_masks.size(1),h,w)
    out_masks = out_masks.view(out_masks.size(0),out_masks.size(1),h,w)

    out_classes = outs[1]
    y_class_perm = true_perm[1]


    out_masks = out_masks.cpu().numpy()
    y_mask_perm = y_mask_perm.cpu().numpy()
    out_classes = out_classes.cpu().numpy()
    y_class_perm = y_class_perm.cpu().numpy()

    out_classes = np.argmax(out_classes,axis=-1)

    return out_masks, out_classes, y_mask_perm, y_class_perm
