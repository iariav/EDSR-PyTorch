import os
import math
from decimal import Decimal
import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.freeze = args.freeze
        self.unfreeze_epoch = args.unfreeze_epoch
        self.optimizer = utility.make_optimizer(args, self.model)
        self.use_amp = True if APEX_AVAILABLE and args.use_amp else False

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        if self.use_amp:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1',
                keep_batchnorm_fp32=None, loss_scale='dynamic',max_loss_scale=64.0)

        self.error_last = 1e8

        if self.freeze:
            def freeze_layer(layer):
                for param in layer.parameters():
                    param.requires_grad = False

            freeze_layer(self.model.model.body)
            freeze_layer(self.model.model.head)
            freeze_layer(self.model.model.tail)
            freeze_layer(self.model.model.head_rgb)
            freeze_layer(self.model.model.body_rgb)
            self.ckp.write_log('\nFroze depth branch parameters.\n')



    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        if self.freeze and epoch > self.unfreeze_epoch:
            def unfreeze_layer(layer):
                for param in layer.parameters():
                    param.requires_grad = True

            unfreeze_layer(self.model.model.body)
            unfreeze_layer(self.model.model.head)
            unfreeze_layer(self.model.model.tail)
            unfreeze_layer(self.model.model.head_rgb)
            unfreeze_layer(self.model.model.body_rgb)
            self.ckp.write_log('\nUnfroze depth branch parameters.\n')

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, rgb,  _,) in enumerate(self.loader_train):
            lr, hr, rgb = self.prepare(lr, hr, rgb)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0, rgb=rgb)
            loss = self.loss(sr, hr)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))


            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        # self.model.save(self.ckp.get_path('model'), epoch, is_best=True)

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, rgb, filename in tqdm(d, ncols=80):
                    lr, hr, rgb = self.prepare(lr, hr, rgb)
                    sr = self.model(lr, idx_scale, rgb)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    curr_rmse = utility.calc_rmse(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    self.ckp.log_rmse[-1, idx_data, idx_scale] += curr_rmse

                    if self.args.save_gt:
                        bc = torch.nn.functional.interpolate(lr, scale_factor=scale, mode='bicubic',align_corners=True)
                        save_list.extend([lr, hr, bc])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                    if self.args.test_only:
                        print('\nRMSE for image \'{}\' and scale {} is: {}.\n'.format(filename[0],scale,curr_rmse))

                self.ckp.log_rmse[-1, idx_data, idx_scale]/= len(d)
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)

                best = self.ckp.log.max(0)
                best_rmse = self.ckp.log_rmse.min(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f}\tRMSE: {:.3f} (Best: {:.3f},{:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.ckp.log_rmse[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best_rmse[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                # self.ckp.write_log('ResScales: {}'.format(self.model._modules['model'].res_scales))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

