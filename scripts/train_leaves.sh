#!/usr/bin/env bash
python train.py -model_name=leaves -max_epoch=10000 -dataset=leaves -num_classes=2 --augment --resize -maxseqlen=20 -gt_maxseqlen=20 -patience_stop=500  -base_model=resnet101 -class_loss_after=-1 -ngpus=2 -batch_size=20 -patience=30 -stop_loss_after=500 --curriculum_learning -min_steps=5 -stop_weight=0.1 -imsize=400 --log_term
