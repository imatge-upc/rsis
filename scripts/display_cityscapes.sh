#!/usr/bin/env bash
python eval.py -model_name=cityscapes -dataset=cityscapes -batch_size=5 -maxseqlen=20  --no_run_coco --display -cityscapes_dir=/gpfs/scratch/bsc31/bsc31429/CityScapes/ --log_term
