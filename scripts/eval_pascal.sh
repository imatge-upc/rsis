#!/usr/bin/env bash
srun-fast --gres=gpu:1,gmem:11G --mem=15G --job-name eval python eval.py -model_name 8 --resize --log_term -class_th 0.0 -stop_th 0.0 --display
