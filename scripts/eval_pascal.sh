#!/usr/bin/env bash
srun-fast --gres=gpu:1,gmem:11G --mem=15G -c 4 --job-name eval python eval.py -model_name 11 --resize --log_term -class_th 0.1
