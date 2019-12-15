import os
import glob
import random
import pickle

from data import common

import numpy as np
import cv2
import imageio
import torch
import torch.utils.data as data
from scipy import ndimage

D_PREFIX = 'Depth'
RGB_PREFIX = 'Image'

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.depth_hr, self.depth_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )

            self.depth_hr, self.images_hr, self.depth_lr = [], [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.depth_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
            for h in list_hr:
                h = h.replace('train_HR', 'train_HR_rgb')
                h = h.replace(D_PREFIX, RGB_PREFIX)
                b = h.replace(self.apath, path_bin)
                h = h.replace('tif','png')
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.depth_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.depth_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.dir_rgb = os.path.join(self.apath, 'HR_rgb')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(np.expand_dims(imageio.imread(img),axis=2), _f)

    def __getitem__(self, idx):
        # print(idx)
        lr, hr, rgb, filename = self._load_file(idx)
        lr, hr, rgb = self.get_patch(lr, hr, rgb)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_t, hr_t = common.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
        rgb_t  = common.np2Tensor(rgb, rgb_range=255.0)
        return lr_t, hr_t, rgb_t[0], filename

    def __len__(self):
        if self.train:
            return len(self.depth_hr) * self.repeat
        else:
            return len(self.depth_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.depth_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.depth_hr[idx]
        f_rgb = f_hr.replace('HR','HR_rgb').replace(D_PREFIX,RGB_PREFIX)
        f_lr = self.depth_lr[self.idx_scale][idx]
        # print(idx)
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
            rgb = imageio.imread(f_rgb)


        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            with open(f_rgb, 'rb') as _f:
                rgb = pickle.load(_f)

        return lr, hr, rgb, filename

    def get_patch(self, lr, hr, rgb):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr, rgb = common.get_patch(
                lr, hr, rgb,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            rgb = np.squeeze(rgb)
            # if len(lr.shape)==2:
            #     lr = np.expand_dims(lr,2)
            # if len(hr.shape) == 2:
            #     hr = np.expand_dims(hr, 2)
            # if len(rgb.shape)==2:
            #     rgb = np.expand_dims(rgb,2)

            if not self.args.no_augment: lr, hr, rgb = common.augment(lr, hr, rgb)
        else:
            rgb = np.squeeze(rgb)
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
            rgb = rgb[0:ih * scale, 0:iw * scale]

        return lr, hr, rgb

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

