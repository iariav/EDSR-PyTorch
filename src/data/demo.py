import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        hr = imageio.imread(self.filelist[idx])
        lr_fname = self.filelist[idx].replace('test_HR','test_LR_bicubic/X2')
        lr = imageio.imread(lr_fname)

        rgb_fname = self.filelist[idx].replace('test_HR', 'test_HR_rgb')
        rgb = imageio.imread(rgb_fname)

        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_t, hr_t = common.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
        rgb_t = common.np2Tensor(rgb, rgb_range=255.0)
        return lr_t, hr_t, rgb_t[0], filename

        # return lr_t, hr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

