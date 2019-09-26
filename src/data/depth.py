import os
from data import srdata
import numpy as np

class depth(srdata.SRData):
    def __init__(self, args, name='depth', train=True, benchmark=False):
        # data_range = [r.split('-') for r in args.data_range.split('/')]
        # if train:
        #     data_range = data_range[0]
        # else:
        #     if args.test_only and len(data_range) == 1:
        #         data_range = data_range[0]
        #     else:
        #         data_range = data_range[1]
        #
        # self.begin, self.end = list(map(lambda x: int(x), data_range))

        if train:
            train_file = os.path.join(args.dir_data, args.data_train[0], 'train.txt')
            self.data_indices = np.loadtxt(train_file,dtype=np.int).flatten()
        else:
            test_file = os.path.join(args.dir_data, args.data_train[0], 'test.txt')
            self.data_indices = np.loadtxt(test_file,dtype=np.int).flatten()
        super(depth, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(depth, self)._scan()
        names_hr = [names_hr[i] for i in self.data_indices]
        for n in range(len(names_lr)):
            names_lr[n] = [names_lr[n][i] for i in self.data_indices]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(depth, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'depth_train_HR')
        self.dir_lr = os.path.join(self.apath, 'depth_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

