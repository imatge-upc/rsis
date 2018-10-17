#!/usr/bin/env bash
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 0 python train.py -model_name 0 --resize --augment -class_loss_after -1 -stop_loss_after -1 -finetune_after -1 -dropout 0.0 --use_box_loss &
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 1 python train.py -model_name 1 --resize --augment -class_loss_after -1 -stop_loss_after -1 -dropout 0 --use_box_loss &
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 2 python train.py -model_name 2 --resize --augment -class_loss_after -1 -stop_loss_after -1 -dropout 0 &
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 4 python train.py -model_name 4 --resize --augment -class_loss_after -1 -stop_loss_after -1 --use_box_loss -dropout 0.5 &
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 5 python train.py -model_name 5 --resize --augment -class_loss_after -1 -stop_loss_after -1 --use_box_loss -dropout 0.5 -finetune_layers 3 &

#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name t python train.py -model_name t --resize --augment --use_box_loss -use_uncertainty_loss --notensorboard --log_term &

#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 6 python train.py -model_name 6 --resize --augment -class_loss_after -1 -stop_loss_after -1 -dropout 0 --use_box_loss -gamma 2 &
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 7 python train.py -model_name 7 --resize --augment -class_loss_after 20 -stop_loss_after -1 -dropout 0 --use_box_loss -uncertainty_loss_after 100 &
#srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 8 python train.py -model_name 8 --resize --augment -class_loss_after 20 -stop_loss_after 120 -dropout 0 --use_box_loss &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 9 python train.py -model_name 9 --resize --augment -class_loss_after 0 -stop_loss_after 0 -dropout 0 --use_box_loss &





