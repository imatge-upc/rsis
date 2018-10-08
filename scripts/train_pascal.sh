#!/usr/bin/env bash
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 1 python train.py -model_name 1 --resize --augment &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 2 python train.py -model_name 2 --resize --augment -hidden_size 256 &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 3 python train.py -model_name 3 --resize --augment -num_lstms 2 &
srun --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name 4 python train.py -model_name 4 --resize --augment -gamma 2.0 &



