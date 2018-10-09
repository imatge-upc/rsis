#!/usr/bin/env bash
srun-fast --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name eval python eval.py -model_name 3 --resize --log_term -class_th 0.0 -stop_th 0.0 --display
