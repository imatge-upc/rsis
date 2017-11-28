#!/usr/bin/env bash
python eval_leaves.py -model_name=leaves -dataset=leaves  -batch_size=5 -maxseqlen=20 --resize -imsize=400 -class_th=0.2 --display --log_term
