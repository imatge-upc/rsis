import numpy as np
import os
import ntpath
import time
import glob
from scipy.misc import imresize
import torchvision.utils as vutils
from operator import itemgetter
from tensorboardX import SummaryWriter


class Visualizer():
    def __init__(self, checkpoints_dir, name):
        self.win_size = 256
        self.name = name
        self.saved = False
        self.checkpoints_dir = checkpoints_dir
        self.ncols = 4

        # remove existing
        for filename in glob.glob(self.checkpoints_dir+"/events*"):
            os.remove(filename)
        self.writer = SummaryWriter(checkpoints_dir)

    def reset(self):
        self.saved = False

    # images: (b, c, 0, 1) array of images
    def image_summary(self, mode, epoch, images, text, normalize):
        images = vutils.make_grid(images, normalize=normalize, scale_each=True)
        in_str = '{}/'+text
        self.writer.add_image(in_str.format(mode), images, epoch)

    # losses: dictionary of error labels and values
    def scalar_summary(self, mode, epoch, **args):
        for k, v in args.items():
            self.writer.add_scalar('{}/{}'.format(mode, k), v, epoch)

        self.writer.export_scalars_to_json("{}/tensorboard_all_scalars.json".format(self.checkpoints_dir))

    def histo_summary(self, model, step):
        """Log a histogram of the tensor of values."""

        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, step)

    def close(self):
        self.writer.close()