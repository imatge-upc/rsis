#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python eval.py -model_name 22 --resize --log_term -class_th 0.5 -stop_th 0.5 --display