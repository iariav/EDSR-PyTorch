## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
import torch
import torch.nn as nn
import numpy as np

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Spatial Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self,channel,kernel_size=3):
        super(SALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_sa = nn.Conv2d(channel,channel,kernel_size, padding=1,groups=channel,stride=2)


    def forward(self, x):
        y = self.conv_sa(x)
        return x * y

## Inject Attention (IAL) Layer
class IALayer(nn.Module):
    def __init__(self,channel,kernel_size=3, reduction=16, use_sa=True, use_ca=True):
        super(IALayer, self).__init__()

        self.use_sa = use_sa
        self.use_ca = use_ca

        # CA
        if self.use_ca:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # feature channel downscale and upscale --> channel weight
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)
            )

        # SA
        if self.use_sa:

            self.conv_sa = nn.Conv2d(channel, channel, kernel_size, padding=1, groups=channel, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if self.use_ca and self.use_sa:
            ca = self.avg_pool(x)
            ca = self.conv_du(ca)
            sa = self.conv_sa(x)
            return self.sigmoid(ca + sa)
        elif self.use_sa:
            sa = self.conv_sa(x)
            return self.sigmoid(sa)
        elif self.use_ca:
            ca = self.avg_pool(x)
            ca = self.conv_du(ca)
            return self.sigmoid(ca)
        else:
            return torch.zeros(1,1)


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias,dilation=2**i,padding = 2**i))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            # if i is not 3: modules_body.append(act)
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class RCAB_RGB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB_RGB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))#,dilation=1,padding = 1))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup_RGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup_RGB, self).__init__()
        modules_body = []
        modules_body = [
            RCAB_RGB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.res_scale = args.res_scale

        ##### Depth branch #####

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.n_colors, args.rgb_range,rgb_mean=[0.5],rgb_std =[1.0])
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.n_colors, args.rgb_range,rgb_mean=[0.5],rgb_std =[1.0], sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        ##### RGB branch #####

        # RGB mean for DIV2K
        self.sub_mean_rgb = common.MeanShift(3, rgb_range=255.0)

        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup_RGB(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.head_AILayer = IALayer(n_feats, kernel_size, reduction,use_sa=args.use_sa, use_ca=args.use_ca)
        modules_AILayers = [IALayer(n_feats, kernel_size, reduction,use_sa=args.use_sa, use_ca=args.use_ca) for _ in range(n_resgroups)]

        # # define tail module
        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]

        # self.add_mean_rgb = common.MeanShift(3, rgb_range=255.0, sign=1)

        self.head_rgb = nn.Sequential(*modules_head)
        self.body_rgb = nn.Sequential(*modules_body)
        self.AILayers = nn.Sequential(*modules_AILayers)
        # self.tail_rgb = nn.Sequential(*modules_tail)

    def forward(self, d, rgb):

        d = self.sub_mean(d)
        d = self.head(d)

        rgb = self.sub_mean_rgb(rgb)
        rgb = self.head_rgb(rgb)

        # A = self.head_AILayer(rgb)
        # min = torch.min(A)
        # max = torch.max(A)
        d += d * self.res_scale * self.head_AILayer(rgb)

        res_d = d
        res_rgb = rgb

        for m in range(len(self.body._modules)):

            d_module   = self.body._modules[str(m)]
            rgb_module = self.body_rgb._modules[str(m)]

            res_d      = d_module(res_d)
            res_rgb    = rgb_module(res_rgb)

            if 'ResidualGroup' in str(d_module):
                # A = self.AILayers[m](res_rgb)
                # min = torch.min(A)
                # max = torch.max(A)
                res_d  += res_d * self.res_scale * self.AILayers[m](res_rgb)

        res_d += d

        d = self.tail(res_d)
        d = self.add_mean(d)

        return d

    def normalizeIm(self, Im):

        Im -= np.min(Im)
        Im /= np.max(Im)


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
