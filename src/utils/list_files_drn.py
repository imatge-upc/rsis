from os import listdir
from os.path import isfile, isdir, join
import math
import numpy as np
import sys
import os
import glob


cityscapes_dir = '/gpfs/scratch/bsc31/bsc31429/CityScapes/'

split = 'val'

image_files = glob.glob(os.path.join(cityscapes_dir, 'leftImg8bit', split, '*', '*.png'))
list_file = open(cityscapes_dir + split  + '_images.txt', 'w')

for i in range(len(image_files)):
  list_file.write(image_files[i].split(cityscapes_dir)[-1] + '\n')
list_file.close()

split = 'test'

image_files = glob.glob(os.path.join(cityscapes_dir, 'leftImg8bit', split, '*', '*.png'))
list_file = open(cityscapes_dir + split  + '_images.txt', 'w')

for i in range(len(image_files)):
  list_file.write(image_files[i].split(cityscapes_dir)[-1] + '\n')
list_file.close()

split = 'train'

image_files = glob.glob(os.path.join(cityscapes_dir, 'leftImg8bit', split, '*', '*.png'))
list_file = open(cityscapes_dir + split  + '_images.txt', 'w')

for i in range(len(image_files)):
  list_file.write(image_files[i].split(cityscapes_dir)[-1] + '\n')
list_file.close()

