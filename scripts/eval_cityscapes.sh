#!/usr/bin/env bash
python eval_cityscapes.py -model_name=cityscapes -dataset=cityscapes -batch_size=5 -maxseqlen=20  --no_run_coco --display --log_term
