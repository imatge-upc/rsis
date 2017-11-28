#!/usr/bin/env bash
python train.py -model_name=cityscapes -dataset=cityscapes -num_classes=9 --augment -maxseqlen=20 -gt_maxseqlen=20  -patience=25 -patience_stop=500 -max_epoch=10000  -class_loss_after=60  -base_model=resnet101 -ngpus=2  -stop_loss_after=100 -batch_size=32  --curriculum_learning  -steps_cl=1 -finetune_after=20 -hidden_size=128 -min_steps=5 --log_term
