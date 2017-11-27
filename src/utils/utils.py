from torch.autograd import Variable
import torch
import os
import numpy as np
import pickle
from collections import OrderedDict

def package_states(gate_info, hidden,gates, T,t, rnn_type='lstm',average=True):
    if rnn_type == 'lstm':
        for i_h,h in enumerate(hidden):
            hidden_state = h[0].cpu().data.numpy()
            cell_state = h[1].cpu().data.numpy()
            if average:
                hidden_state = np.mean(hidden_state,axis=1)
                cell_state = np.mean(cell_state,axis=1)
            if i_h not in gate_info['hidden'].keys():
                shape = [T]
                for s in hidden_state.shape:
                    shape.append(s)
                gate_info['hidden'][i_h] = np.zeros((shape))
                gate_info['cell'][i_h] = np.zeros((shape))
            gate_info['hidden'][i_h][t] = hidden_state
            gate_info['cell'][i_h][t] = cell_state


        for i_g,gate in enumerate(gates):
            input_gate = gate[0].cpu().data.numpy()
            forget_gate = gate[1].cpu().data.numpy()
            if average:
                input_gate = np.mean(hidden_state,axis=1)
                forget_gate = np.mean(forget_gate,axis=1)
            if i_g not in gate_info['forget'].keys():
                shape = [T]
                for s in forget_gate.shape:
                    shape.append(s)
                gate_info['input'][i_g] = np.zeros((shape))
                gate_info['forget'][i_g] = np.zeros((shape))
            gate_info['forget'][i_g][t] = forget_gate
            gate_info['input'][i_g][t] = input_gate

    else:
        for i_h,h in enumerate(hidden):
            hidden_state = h.cpu().data.numpy()
            if average:
                hidden_state = np.mean(hidden_state,axis=1)
            if i_h not in gate_info['hidden'].keys():
                shape = [T]
                for s in hidden_state.shape:
                    shape.append(s)
                gate_info['hidden'][i_h] = np.zeros((shape))
            gate_info['hidden'][i_h][t] = hidden_state

        for i_g, gate in enumerate(gates):
            reset_gate = gate[0].cpu().data.numpy()
            update_gate = gate[1].cpu().data.numpy()
            if average:
                reset_gate = np.mean(reset_gate,axis=1)
                update_gate = np.mean(update_gate,axis=1)
            if i_g not in gate_info['update'].keys():
                shape = [T]
                for s in update_gate.shape:
                    shape.append(s)
                gate_info['update'][i_g] = np.zeros((shape))
                gate_info['reset'][i_g] = np.zeros((shape))
            gate_info['update'][i_g][t] = update_gate
            gate_info['reset'][i_g][t] = reset_gate

    return gate_info

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def adjust_learning_rate(step, optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every step epochs"""
    new_lr = lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

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

def get_base_params(args, model):
    b = []

    if 'drn' in args.base_model:
        b.append(model.base)
    elif 'vgg' in args.base_model:
        b.append(model.base.features)
    else:
        b.append(model.base.conv1)
        b.append(model.base.bn1)
        b.append(model.base.layer1)
        b.append(model.base.layer2)
        b.append(model.base.layer3)
        b.append(model.base.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_skip_params(model):
    b = []

    b.append(model.sk1.parameters())
    b.append(model.sk2.parameters())
    b.append(model.sk3.parameters())
    b.append(model.sk4.parameters())
    b.append(model.sk5.parameters())
    b.append(model.bn1.parameters())
    b.append(model.bn2.parameters())
    b.append(model.bn3.parameters())
    b.append(model.bn4.parameters())
    b.append(model.bn5.parameters())


    for j in range(len(b)):
        for i in b[j]:
            yield i

def merge_params(params):
    for j in range(len(params)):
        for i in params[j]:
            yield i

def get_optimizer(optim_name, lr, parameters, weight_decay = 0, momentum = 0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                lr=lr, weight_decay = weight_decay,
                                momentum = momentum)
    elif optim_name =='adam':
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    elif optim_name =='rmsprop':
        opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    return opt

def save_checkpoint(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(), os.path.join('../models',args.model_name,'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../models',args.model_name,'decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join('../models',args.model_name,'enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join('../models',args.model_name,'dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../models',args.model_name,'args.pkl'),'wb'))

def load_checkpoint(model_name):
    encoder_dict = torch.load(os.path.join('../models',model_name,'encoder.pt'))
    decoder_dict = torch.load(os.path.join('../models',model_name,'decoder.pt'))
    enc_opt_dict = torch.load(os.path.join('../models',model_name,'enc_opt.pt'))
    dec_opt_dict = torch.load(os.path.join('../models',model_name,'dec_opt.pt'))
    # save parameters for future use
    args = pickle.load(open(os.path.join('../models',model_name,'args.pkl'),'rb'))

    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args

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
    elif model_name in ['drn_c_26','drn_c_42']:
        skip_dims_in = [512,512,512,256,128]
    elif model_name == 'drn_c_58':
        skip_dims_in = [512,512,2048,1024,512]
    elif model_name == 'drn_d_22':
        skip_dims_in = [512,256,128,64,32]

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

    elot['xentr'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='BCE Loss',
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

    if args.use_feedback:
        mviz_prev = {}
        for i in range(args.maxseqlen):
            mviz_prev[i] = viz.heatmap(X=np.zeros((args.imsize,args.imsize)),
                                       opts=dict(title='Previous mask t'))
    else:
        mviz_prev = None

    image_lot = viz.image(np.ones((3,args.imsize,args.imsize)),
                        opts=dict(title='image'))


    return lot, elot, mviz_pred, mviz_true, mviz_prev, image_lot

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
    if args.use_feedback:
        prev_masks = outs[2]
        prev_masks = prev_masks.view(prev_masks.size(0),prev_masks.size(1),h,w)
        prev_masks = prev_masks.cpu().numpy()

        return out_masks, out_classes, y_mask_perm, y_class_perm, prev_masks
    else:
        return out_masks, out_classes, y_mask_perm, y_class_perm, None
