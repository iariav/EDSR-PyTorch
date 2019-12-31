from model import common_rnan
from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return RNAN(args)
### RNAN
class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_ResGroup, self).__init__()
        modules_body = []
        modules_body.append(common.ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class _NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_NLResGroup, self).__init__()
        modules_body = []
        modules_body.append(common.NLResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class RNAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RNAN, self).__init__()
        
        n_resgroup = args.n_resgroups
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        # define body module
        # modules_body_nl_low = [
        #     _NLResGroup(
        #         conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
        modules_body1 = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) \
            for _ in range(n_resgroup - 2)]
        # modules_body_nl_high = [
        #     _NLResGroup(
        #         conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
        modules_body1.append(conv(n_feats, n_feats, kernel_size))

        modules_body2 = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) \
            for _ in range(n_resgroup - 2)]
        modules_body2.append(conv(n_feats, n_feats, kernel_size))

        # define tail module       


        self.head = nn.Sequential(*modules_head)
        # self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body1 = nn.Sequential(*modules_body1)
        self.body2 = nn.Sequential(*modules_body2)
        # self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        # self.tail = nn.Sequential(*modules_tail)

        # Up-sampling net
        if self.scale == 2 or self.scale == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * self.scale * self.scale, kernel_size, padding=(kernel_size - 1) // 2,
                          stride=1),
                nn.PixelShuffle(self.scale)
                # nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif self.scale == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                # nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

        ##### RGB branch #####

        modules_rgb = [conv(3, n_feats, kernel_size=1),
                       conv(n_feats, n_feats, kernel_size=1, padding=0, stride=1),
                       nn.ReLU(True)]

        self.rgb_os = nn.Sequential(*modules_rgb)

        modules_rgb_downsize = [conv(n_feats, n_feats, kernel_size=1),
                                conv(n_feats, n_feats, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                     stride=2),
                                nn.ReLU(True)]

        self.rgb_ds = nn.Sequential(*modules_rgb_downsize)

        self.fuse_conv = conv(n_feats * 2, n_feats, kernel_size=1)

        self.AILayer = common_rnan.NLMaskBranchDownUp(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.AILayer_ds = common_rnan.NLMaskBranchDownUp(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.tail_conv = conv(n_feats, args.n_colors, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1)



    def forward(self, x, rgb):

        rgb_os = self.rgb_os(rgb)
        rgb_ds = self.rgb_ds(rgb_os)

        AI_ds = self.AILayer_ds(rgb_ds)
        AI = self.AILayer(rgb_os)

        feats_shallow = self.head(x)
        feats_shallow = feats_shallow.mul(AI_ds)
        res = self.body1(feats_shallow)
        res += feats_shallow

        res_up = self.UPNet(res)
        res_up = res_up.mul(AI)
        res_up_out = self.body2(res_up)

        res_up_out += res_up

        out = self.tail_conv(res_up_out)

        return out


    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

