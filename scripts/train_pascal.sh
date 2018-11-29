#!/usr/bin/env bash

srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name coordconv python train.py -model_name coordconv  \
--resize --augment -class_loss_after 20 -stop_loss_after 100 -dropout 0.5 --coordconv -batch_size 20 &

srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 5 python train.py -model_name conv  \
--resize --augment -class_loss_after 20 -stop_loss_after 100 -dropout 0.5
