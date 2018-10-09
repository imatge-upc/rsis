#!/usr/bin/env bash
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 0 python train.py -model_name 0 --resize --augment -class_loss_after -1 -stop_loss_after -1 -finetune_after -1 &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 1 python train.py -model_name 1 --resize --augment -class_loss_after -1 -stop_loss_after -1 &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 2 python train.py -model_name 2 --resize --augment -class_loss_after 20 -stop_loss_after 100 &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 3 python train.py -model_name 3 --resize --augment -class_loss_after -1 -stop_loss_after -1 --use_box_loss &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 4 python train.py -model_name 4 --resize --augment -class_loss_after 20 -stop_loss_after 100 --use_box_loss &
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 3 python train.py -model_name 3 --resize --augment -class_loss_after -1 -stop_loss_after -1 --use_box_loss --notensorboard --log_term &



