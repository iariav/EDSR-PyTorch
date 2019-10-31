import os
from data import srdata
import numpy as np

class dsm(srdata.SRData):
    def __init__(self, args, name='depth', train=True, benchmark=False):

        if train:
            train_file = os.path.join(args.dir_data, args.data_train[0], 'train.txt')
            self.data_indices = np.loadtxt(train_file,dtype=np.int).flatten()
        else:
            test_file = os.path.join(args.dir_data, args.data_train[0], 'test.txt')
            self.data_indices = np.loadtxt(test_file,dtype=np.int).flatten()
        super(dsm, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(dsm, self)._scan()
        names_hr = [names_hr[i] for i in self.data_indices]
        for n in range(len(names_lr)):
            names_lr[n] = [names_lr[n][i] for i in self.data_indices]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(dsm, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'dsm_train_HR')
        self.dir_lr = os.path.join(self.apath, 'dsm_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

