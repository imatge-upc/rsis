import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='RIASS')

    ## TRAINING parameters ##
    parser.add_argument('--resume', dest='resume',action='store_true',
                        help=('whether to resume training an existing model '
                        '(the one with name model_name will be used)'))
    parser.set_defaults(resume=False)
    # set epoch_resume if you want flags --finetune_after and --update_encoder to be properly
    # activated (eg if you stop training for whatever reason at epoch 15, set epoch_resume to 15)
    parser.add_argument('-epoch_resume', dest='epoch_resume',default= 0,type=int,
                        help=('set epoch_resume if you want flags '
                        '--finetune_after and --update_encoder to be properly '
                        'activated (eg if you stop training for whatever reason '
                        'at epoch 15, set epoch_resume to 15)'))
    parser.add_argument('-seed', dest='seed',default = 123, type=int)
    parser.add_argument('-batch_size', dest='batch_size', default = 32, type=int)
    parser.add_argument('-lr', dest='lr', default = 1e-3,type=float)
    parser.add_argument('-momentum', dest='momentum', default =0.9,type=float)
    parser.add_argument('-adjust_step', dest='adjust_step', default =100,type=int)
    parser.add_argument('-optim', dest='optim', default = 'adam',
                        choices=['adam','sgd','rmsprop'])
    parser.add_argument('-maxseqlen', dest='maxseqlen', default = 5, type=int)
    parser.add_argument('-gt_maxseqlen', dest='gt_maxseqlen', default = 20, type=int)
    parser.add_argument('-clip', dest='clip', default = -1, type=float)
    parser.add_argument('-best_val_loss', dest='best_val_loss', default = 1000, type=float)
    parser.add_argument('--predict_eos_mask',dest='predict_eos_mask', action='store_true')
    parser.set_defaults(predict_eos_mask=False)
    parser.add_argument('--impose_order', dest='impose_order', action='store_true')
    parser.set_defaults(impose_order=False)
    parser.add_argument('--balanced_bce', dest='balanced_bce', action='store_true')
    parser.set_defaults(balanced_bce=False)
    parser.add_argument('--balance_class_loss', dest='balance_class_loss', action='store_true')
    parser.set_defaults(balance_class_loss=False)
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.set_defaults(crop=False)
    parser.add_argument('--unique_branch', dest='unique_branch', action='store_true')
    parser.set_defaults(unique_branch=False)
    parser.add_argument('--smooth_curves',dest='smooth_curves', action='store_true')
    parser.set_defaults(smooth_curves=False)

    # base model fine tuning
    parser.add_argument('-finetune_after', dest='finetune_after', default = 3000, type=int,
                        help=('epoch number to start finetuning. set -1 to not finetune.'
                        'there is a patience term that can allow starting to fine tune '
                        'earlier (does not apply if value is -1)'))
    parser.add_argument('-lr_cnn', dest='lr_cnn', default = 5e-5,type=float)
    parser.add_argument('-optim_cnn', dest='optim_cnn', default = 'adam',
                        choices=['adam','sgd','rmsprop'])
    parser.add_argument('--update_encoder', dest='update_encoder', action='store_true',
                        help='used in sync with finetune_after. no need to activate.')
    parser.set_defaults(update_encoder=False)
    parser.add_argument('-weight_decay', dest='weight_decay', default = 0, type=float)
    parser.add_argument('-weight_decay_cnn', dest='weight_decay_cnn', default = 0, type=float)
    parser.add_argument('--transfer',dest='transfer', action='store_true')
    parser.set_defaults(transfer=False)
    parser.add_argument('-transfer_from', dest='transfer_from', default = '15_')
    parser.add_argument('--curriculum_learning',dest='curriculum_learning', action='store_true')
    parser.set_defaults(curriculum_learning=False)
    parser.add_argument('-steps_cl', dest='steps_cl', default=1, type=int)
    parser.add_argument('-min_steps', dest='min_steps', default=1, type=int)
    parser.add_argument('-min_delta', dest='min_delta', default=0.0, type=float)

    # feedback
    parser.add_argument('-feed_preds_after', dest='feed_preds_after', default = -1,type=int,
                        help='start feeding true predictions after N epochs. -1 to never do it.')
    parser.add_argument('--feed_prediction', dest='feed_prediction', action='store_true')
    parser.set_defaults(feed_prediction=False)
    parser.add_argument('--use_feedback', dest='use_feedback', action='store_true',
                        help='whether to use the previous mask as input to the network.')
    parser.set_defaults(use_feedback=False)

    # Cross entropy loss
    parser.add_argument('-class_loss_after', dest='class_loss_after', default=3000, type=int,
                        help=('epoch number to start training the classification loss. '
                        'set to -1 to not do it. A patience term can allow to start '
                        'training with this loss (does not apply if value is -1)'))
    parser.add_argument('--use_class_loss', dest='use_class_loss', action='store_true')
    parser.set_defaults(use_class_loss=False)
    parser.add_argument('-stop_loss_after', dest='stop_loss_after', default = 3000, type=int,
                        help=('epoch number to start training the stopping loss. '
                        'set to -1 to not do it. A patience term can allow to start '
                        'training with this loss (does not apply if value is -1)'))
    parser.add_argument('--use_stop_loss', dest='use_stop_loss', action = 'store_true')
    parser.set_defaults(use_stop_loss=False)

    # stopping criterion
    parser.add_argument('-patience', dest='patience', default = 5, type=int,
                        help=('patience term to activate flags such as '
                        'use_class_loss, feed_prediction and update_encoder if '
                        'their matching vars are not -1'))
    parser.add_argument('-patience_stop', dest='patience_stop', default = 20, type=int,
                        help='patience to stop training.')
    parser.add_argument('-max_epoch', dest='max_epoch', default = 1000, type=int)

    # visualization and logging
    parser.add_argument('-print_every', dest='print_every', default = 10, type=int)
    parser.add_argument('--log_term', dest='log_term', action='store_true',
                        help='if activated, will show logs in stdout instead of log file.')
    parser.set_defaults(log_term=False)
    parser.add_argument('--visdom', dest='visdom', action='store_true')
    parser.set_defaults(visdom=False)
    parser.add_argument('-port',dest='port',default=1061, type=int, help='visdom port')
    parser.add_argument('-server',dest='server',default='http://c2', help='visdom server')

    # loss weights
    parser.add_argument('-class_weight',dest='class_weight',default=0.1, type=float)
    parser.add_argument('-iou_weight',dest='iou_weight',default=1.0, type=float)
    parser.add_argument('-bce_weight',dest='bce_weight',default=0.0, type=float)
    parser.add_argument('-stop_weight',dest='stop_weight',default=1.0, type=float)
    parser.add_argument('-stop_balance_weight',dest='stop_balance_weight',default=0.5, type=float)
    # augmentation
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.set_defaults(augment=False)
    parser.add_argument('-rotation', dest='rotation', default = 10, type=int)
    parser.add_argument('-translation', dest='translation', default = 0.1, type=float)
    parser.add_argument('-shear', dest='shear', default = 0.1, type=float)
    parser.add_argument('-zoom', dest='zoom', default = 0.7, type=float)

    # GPU
    parser.add_argument('--cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('-ngpus', dest='ngpus', default=1,type=int)

    # model parameters
    parser.add_argument('--wideresnet',dest='wideresnet',action='store_true')
    parser.set_defaults(wideresnet=False)
    parser.add_argument('-base_model', dest='base_model', default = 'resnet50',
                        choices=['resnet101','resnet50','resnet34','vgg16',
                        'drn_d_22','drn_d_38'])
    parser.add_argument('-skip_mode', dest='skip_mode', default = 'concat',
                        choices=['sum','concat','mul','none'])
    parser.add_argument('-model_name', dest='model_name', default='model')
    parser.add_argument('-log_file', dest='log_file', default='train.log')
    parser.add_argument('-hidden_size', dest='hidden_size', default = 64, type=int)
    parser.add_argument('-hidden_conv_skip', dest='hidden_conv_skip', default = 64, type=int)
    parser.add_argument('-D', dest='D', default = 32, type=int)
    parser.add_argument('--limit_width',dest='limit_width', action='store_true')
    parser.set_defaults(limit_width=False)
    parser.add_argument('--drn_seg', dest='drn_seg', action='store_true')
    parser.set_defaults(drn_seg=False)
    parser.add_argument('-kernel_size', dest='kernel_size', default = 3, type=int)
    parser.add_argument('-dropout', dest='dropout', default = 0.0, type=float)
    parser.add_argument('-dropout_stop', dest='dropout_stop', default = 0.5, type=float)
    parser.add_argument('-dropout_cls', dest='dropout_cls', default = 0.0, type=float)
    parser.add_argument('-nconvlstm', dest='nconvlstm', default = 5, type=int,choices=[1,2,3,4,5])
    parser.add_argument('-conv_start', dest='conv_start', default = 0, type=int,choices=[0,1,2,3,4])
    parser.add_argument('-rnn_type', dest='rnn_type', default = 'lstm', choices = ['lstm','gru'])
    parser.add_argument('--insnorm',dest='batchnorm', action='store_false')
    parser.set_defaults(batchnorm=True)
    parser.add_argument('--multiscale',dest='multiscale', action='store_true')
    parser.set_defaults(multiscale=False)
    parser.add_argument('--use_ss_model',dest='use_ss_model', action='store_true')
    parser.set_defaults(use_ss_model=False)
    parser.add_argument('-ss_model_path',dest='ss_model_path', default='../models/semseg/encoder.pt')
    parser.add_argument('--use_coco_weights',dest='use_coco_weights', action='store_true')
    parser.set_defaults(use_coco_weights=False)
    parser.add_argument('-deeplab_coco_model',dest='deeplab_coco_model',
                        default='../models/deeplab_voc12.pth')
    parser.add_argument('-seg_checkpoint_path', dest='seg_checkpoint_path',
                        default='/home/bsc31/bsc31429/.torch/models/drn_d_22_cityscapes.pth')



    # dataset parameters
    parser.add_argument('-imscale',dest='imscale', default=256, type=int)
    parser.add_argument('-imsize',dest='imsize', default=256, type=int)
    parser.add_argument('--resize',dest='resize', action='store_true')
    parser.set_defaults(resize=False)
    parser.add_argument('-num_classes', dest='num_classes', default = 21, type=int)
    parser.add_argument('-dataset', dest='dataset', default = 'pascal',choices=['pascal','coco','cityscapes', 'leaves'])
    parser.add_argument('-pascal_dir', dest='pascal_dir',
                        default = '/work/asalvador/dev/data/dettention/datasets/VOCPlus2/')
    parser.add_argument('-coco_dir', dest='coco_dir',
                        default = '/work/asalvador/dev/data/dettention/datasets/COCO/')
    parser.add_argument('-cityscapes_dir', dest='cityscapes_dir',
                        default='/gpfs/scratch/bsc31/bsc31429/CityScapes/')
    parser.add_argument('-leaves_dir', dest='leaves_dir',
                        default='/gpfs/scratch/bsc31/bsc31429/LeavesDataset/A1/')
    parser.add_argument('-leaves_test_dir', dest='leaves_test_dir',
                        default = '/gpfs/scratch/bsc31/bsc31429/CVPPP2014_LSC_testing_data/A1/')


    parser.add_argument('-num_workers', dest='num_workers', default = 4, type=int)
    parser.add_argument('--small_cityscapes',dest='small_cityscapes', action='store_true')
    parser.set_defaults(small_cityscapes=False)

    # testing
    parser.add_argument('-eval_split',dest='eval_split', default='val')
    parser.add_argument('-mask_th',dest='mask_th', default=0.5, type=float)
    parser.add_argument('-stop_th',dest='stop_th', default=0.5, type=float)
    parser.add_argument('-max_dets',dest='max_dets', default=100, type=int)
    parser.add_argument('-min_size',dest='min_size', default=0.001, type=float)
    parser.add_argument('-cat_id',dest='cat_id', default=-1,type=int)
    parser.add_argument('--ignore_cats',dest='use_cats', action='store_false')
    parser.add_argument('--use_gt_cats',dest='use_gt_cats', action='store_true')
    parser.add_argument('--use_gt_masks',dest='use_gt_masks', action='store_true')
    parser.add_argument('--use_gt_stop',dest='use_gt_stop', action='store_true')
    parser.add_argument('-class_th',dest='class_th', default=0.5, type=float)
    parser.add_argument('--display', dest='display', action='store_true')
    parser.add_argument('--no_display_text', dest='no_display_text', action='store_true')
    parser.add_argument('--all_classes',dest='all_classes', action='store_true')
    parser.add_argument('--keep_biggest_blob',dest='keep_biggest_blob', action='store_true')
    parser.add_argument('--no_run_coco_eval',dest='no_run_coco_eval', action='store_true')
    parser.add_argument('--flip_in_eval', dest='flip_in_eval', action='store_true')
    parser.add_argument('--save_gates', dest='save_gates', action='store_true')
    parser.add_argument('--extract_raw', dest='extract_raw', action='store_true')
    parser.add_argument('--display_route', dest='display_route', action='store_true')
    parser.add_argument('--average_gates', dest='average_gates', action='store_true')
    parser.set_defaults(display=False)
    parser.set_defaults(display_route=False)
    parser.set_defaults(use_cats=True)
    parser.set_defaults(all_classes=False)
    parser.set_defaults(keep_biggest_blob=False)
    parser.set_defaults(no_display_text=False)
    parser.set_defaults(flip_in_eval=False)
    parser.set_defaults(save_gates=False)

    parser.set_defaults(average_gates=False)
    parser.set_defaults(extract_raw=False)
    parser.set_defaults(use_gt_cats=False)
    parser.set_defaults(use_gt_masks=False)
    parser.set_defaults(use_gt_stop=False)
    return parser

if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()
