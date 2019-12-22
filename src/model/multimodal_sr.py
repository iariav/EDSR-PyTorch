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


class RDN(nn.Module):
    def __init__(self, channel, growth_rate, rdb_number, upscale_factor):
        super(RDN, self).__init__()
        self.SFF1 = nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.SFF2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.RDB1 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB2 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB3 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.GFF1 = nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, padding=0)
        self.GFF2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.upconv = nn.Conv2d(in_channels=64, out_channels=(64 * upscale_factor * upscale_factor), kernel_size=3,
                                padding=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=channel, kernel_size=3, padding=1)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def forward(self, x):
        f_ = self.SFF1(x)
        f_0 = self.SFF2(f_)
        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_D = torch.cat((f_1, f_2, f_3), 1)
        f_1x1 = self.GFF1(f_D)
        f_GF = self.GFF2(f_1x1)
        f_DF = f_GF + f_
        f_upconv = self.upconv(f_DF)
        f_upscale = self.pixelshuffle(f_upconv)
        f_conv2 = self.conv2(f_upscale)
        return f_conv2


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, nb_layers, input_dim, growth_rate):
        super(RDB, self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers, input_dim, growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels=input_dim + nb_layers * growth_rate, \
                                 out_channels=growth_rate, \
                                 kernel_size=1, \
                                 stride=1, \
                                 padding=0)

    def _make_layer(self, nb_layers, input_dim, growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(input_dim + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        out = self.conv1x1(out)
        return out + x

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
            modules_body.append(act)
            # if i is not 3: modules_body.append(act)
            # if i == 0: modules_body.append(act)
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
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks,out_channels):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        # modules_body.append(conv(in_channels=n_feat, out_channels=int(n_feat/2), kernel_size=kernel_size))
        self.body = nn.Sequential(*modules_body)
        # self.conv = conv(in_channels=n_feat, out_channels=int(n_feat/2), kernel_size=kernel_size) # for concat
        convs = []
        convs = [conv(in_channels=n_feat, out_channels=int(n_feat), kernel_size=1),
                    conv(in_channels=n_feat, out_channels=out_channels, kernel_size=kernel_size)]# for add
        self.conv = nn.Sequential(*convs)

    def forward(self, x):
        res = self.body(x)
        res += x
        res = self.conv(res)
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
        self.use_attention = args.use_attention
        self.scale = scale

        ##### Depth branch #####

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.n_colors, args.rgb_range,rgb_mean=[0.5],rgb_std =[1.0])
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [ # for concat
            ResidualGroup(conv, n_feats*2, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks,out_channels=int(n_feats)),
            ResidualGroup(conv, n_feats*2, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, out_channels=int(n_feats*2))]

        # modules_body = [ # for add
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats*2, n_feats*2, kernel_size)) # for concat
        # modules_body.append(conv(n_feats , n_feats , kernel_size))  # for add

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats*2, act=False), # for concat
            conv(n_feats*2, args.n_colors, kernel_size)]
        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats , act=False), # for add
        #     conv(n_feats , args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.n_colors, args.rgb_range,rgb_mean=[0.5],rgb_std =[1.0], sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        ##### RGB branch #####

        # RGB mean for DIV2K
        self.sub_mean_rgb = common.MeanShift(3, 255.0)

        # define head module
        modules_head = [conv(3, n_feats, kernel_size),
                        conv(n_feats,n_feats,kernel_size=1),
                        nn.ReLU(True)]

        # define body module
        modules_body = [conv(n_feats,n_feats,kernel_size=1),
                        conv(n_feats, n_feats, kernel_size=3),
                        nn.ReLU(True)]

        modules_tail = [conv(n_feats, n_feats, kernel_size=1),
                        conv(n_feats, n_feats*2, kernel_size=1),
                        nn.ReLU(True)]
        # modules_body = [
        #     ResidualGroup_RGB(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]

        # modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.head_AILayer = IALayer(n_feats, kernel_size, reduction,use_sa=args.use_sa, use_ca=args.use_ca)
        modules_AILayers = [IALayer(n_feats, kernel_size, reduction,use_sa=args.use_sa, use_ca=args.use_ca) for _ in range(n_resgroups)]

        # # define tail module
        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]

        # self.add_mean_rgb = common.MeanShift(3, rgb_range=255.0, sign=1)

        self.head_rgb = nn.Sequential(*modules_head)
        self.body_rgb = nn.Sequential(*modules_body)
        self.tail_rgb = nn.Sequential(*modules_tail)
        self.AILayers = nn.Sequential(*modules_AILayers)
        # self.tail_rgb = nn.Sequential(*modules_tail)

        # self.res_scales = torch.nn.Parameter(torch.full((n_resgroups+1,1),fill_value=0.5),requires_grad=False)

    def forward(self, d, rgb):

        d = self.sub_mean(d)
        d = self.head(d)

        rgb = self.sub_mean_rgb(rgb)
        # min_rgb = torch.min(rgb)
        # max_rgb = torch.max(rgb)
        # rgb -= min_rgb
        # rgb /= (max_rgb - min_rgb)
        rgb = self.head_rgb(rgb)

        if self.use_attention:
            AI = self.head_AILayer(rgb)
            alpha = 0.5#self.res_scales[0]
            # d = d.mul(alpha) + AI.mul(1 - alpha)
            d = torch.cat((d,d*AI),1)
        else:
            AI = nn.functional.interpolate(rgb, scale_factor=float(1/self.scale), mode='bilinear', align_corners=True)
            d = torch.cat((d, AI), 1)

        res_d = d
        res_rgb = rgb

        for m in range(len(self.body._modules)):

            d_module   = self.body._modules[str(m)]
            res_d = d_module(res_d)

            if m==0:#(len(self.body._modules)/2):
                # rgb_module = self.body_rgb._modules[str(m)]
                # res_rgb    = rgb_module(res_rgb)
                res_rgb    = self.body_rgb(res_rgb)
                # A = self.AILayers[m](res_rgb)
                # min = torch.min(A)
                # max = torch.max(A)
                if self.use_attention:
                    AI = self.AILayers[m](res_rgb)
                    alpha = 0.5#self.res_scales[m + 1]
                    res_d = torch.cat((res_d, res_d*AI), 1)
                    # res_d = res_d.mul(alpha) + AI.mul(1 - alpha)
                else:
                    AI = nn.functional.interpolate(res_rgb, scale_factor=float(1/self.scale), mode='bilinear',
                                                   align_corners=True)
                    res_d = torch.cat((res_d, AI), 1)

        # res_rgb = self.tail_rgb(res_rgb)
        # AI = self.AILayers[-1](res_rgb)
        # res_d = res_d.mul(alpha) + AI.mul(1 - alpha)
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
