#!/usr/bin/env bash
srun --gres=gpu:1,gmem:11G --mem=15G python train.py -model_name rsis-pascal --resize -pascal_dir /work/asalvador/dev/data/dettention/datasets/VOCAug --log_term