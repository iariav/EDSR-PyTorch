## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common
import torch
import torch.nn as nn
import numpy as np
import math

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

            # self.conv_sa = nn.Conv2d(channel, channel, kernel_size, padding=(kernel_size-1)//2, groups=channel, stride=1)

            self.conv_sa = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True)
            )

            # self.conv_sa1 = nn.Sequential(
            #         nn.Conv2d(channel, channel//2, kernel_size=[1,9], padding=2, stride=1),
            #         nn.Conv2d(channel// 2, 1, kernel_size=[9, 1], padding=2, stride=1)
            # )
            #
            # self.conv_sa2 = nn.Sequential(
            #         nn.Conv2d(channel, channel//2, kernel_size=[9,1], padding=2, stride=1),
            #         nn.Conv2d(channel// 2, 1, kernel_size=[1, 9], padding=2, stride=1)
            # )


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if self.use_ca and self.use_sa:
            ca = self.avg_pool(x)
            ca = self.conv_du(ca)
            # sa = self.conv_sa1(x) + self.conv_sa2(x)
            sa = self.conv_sa(x)
            return self.sigmoid(ca + sa)
        elif self.use_sa:
            # sa = self.conv_sa1(x) + self.conv_sa2(x)
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
        for i in range(4):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias,dilation=2**i,padding = 2**i))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
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
        # modules_body.append(conv(in_channels=n_feat, out_channels=int(n_feat/2), kernel_size=kernel_size))
        self.body = nn.Sequential(*modules_body)
        # self.conv = conv(in_channels=n_feat, out_channels=int(n_feat/2), kernel_size=kernel_size) # for concat
        self.conv = conv(in_channels=n_feat, out_channels=int(n_feat), kernel_size=kernel_size) # for add

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.conv(res)

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.res_scale = args.res_scale
        self.use_attention = args.use_attention
        self.scale = scale

        n_resgroups = args.n_resgroups*int(math.log(scale, 2))

        ##### Depth branch #####

        # define head module
        self.SFENet1 = nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size - 1) // 2, stride=1)

        # define body module

        self.resgroups = nn.ModuleList()
        for i in range(2):
            self.resgroups.append(
                ResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
            )

        # self.resgroups.append(conv(n_feats, n_feats, kernel_size)) # for concat

        # for scale 4

        self.resgroups_s4 = nn.ModuleList()
        for i in range(2):
            self.resgroups_s4.append(
                ResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
            )

        # self.resgroups_s4.append(conv(n_feats, n_feats, kernel_size))  # for concat

        # for scale 8

        self.resgroups_s8 = nn.ModuleList()
        for i in range(2):
            self.resgroups_s8.append(
                ResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
            )


        # Up-sampling net

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
            nn.PixelShuffle(2)
        ])


        self.UPNet_s4 = nn.Sequential(*[
            nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2,
                      stride=1),
            nn.PixelShuffle(2)
        ])

        self.UPNet_s8 = nn.Sequential(*[
            nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2,
                      stride=1),
            nn.PixelShuffle(2)
        ])

        ##### RGB branch #####

        ### RGB BRANCH ###
        modules_rgb = [conv(3, n_feats, kernel_size=1),
                       conv(n_feats, n_feats, kernel_size=1, padding=0, stride=1),
                       nn.ReLU(True)]

        self.rgb_os = nn.Sequential(*modules_rgb)

        modules_rgb_downsize = [conv(n_feats, n_feats, kernel_size=1),
                                conv(n_feats, n_feats, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=2),
                                nn.ReLU(True)]

        self.rgb_ds = nn.Sequential(*modules_rgb_downsize)


        self.fuse_conv = conv(n_feats * 2, n_feats, kernel_size=1)


        self.AILayer = IALayer(n_feats, kernel_size, reduction, use_sa=args.use_sa, use_ca=args.use_ca)
        self.AILayer_ds = IALayer(n_feats, kernel_size, reduction, use_sa=args.use_sa, use_ca=args.use_ca)

        ### RGB BRANCH scale 4 ###
        modules_rgb_s4 = [conv(n_feats, n_feats, kernel_size=1),
                       conv(n_feats, n_feats, kernel_size=1, padding=0, stride=1),
                       nn.ReLU(True)]

        self.rgb_os_s4 = nn.Sequential(*modules_rgb_s4)

        modules_rgb_downsize_s4 = [conv(n_feats, n_feats, kernel_size=1),
                                conv(n_feats, n_feats, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                     stride=2),
                                nn.ReLU(True)]

        self.rgb_ds_s4 = nn.Sequential(*modules_rgb_downsize_s4)

        self.fuse_conv_s4 = conv(n_feats * 2, n_feats, kernel_size=1)

        self.AILayer_s4 = IALayer(n_feats, kernel_size, reduction, use_sa=args.use_sa, use_ca=args.use_ca)
        self.AILayer_ds_s4 = IALayer(n_feats, kernel_size, reduction, use_sa=args.use_sa, use_ca=args.use_ca)

        ### RGB BRANCH scale 8 ###
        modules_rgb_s8 = [conv(n_feats, n_feats, kernel_size=1),
                          conv(n_feats, n_feats, kernel_size=1, padding=0, stride=1),
                          nn.ReLU(True)]

        self.rgb_os_s8 = nn.Sequential(*modules_rgb_s8)

        modules_rgb_downsize_s8 = [conv(n_feats, n_feats, kernel_size=1),
                                   conv(n_feats, n_feats, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                        stride=2),
                                   nn.ReLU(True)]

        self.rgb_ds_s8 = nn.Sequential(*modules_rgb_downsize_s8)

        self.fuse_conv_s8 = conv(n_feats * 2, n_feats, kernel_size=1)

        self.AILayer_s8 = IALayer(n_feats, kernel_size, reduction, use_sa=args.use_sa, use_ca=args.use_ca)
        self.AILayer_ds_s8 = IALayer(n_feats, kernel_size, reduction, use_sa=args.use_sa, use_ca=args.use_ca)

        self.tail_conv = conv(n_feats, args.n_colors, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=1)

    def forward(self, d, rgb):

        ## scale 2
        rgb_os = self.rgb_os(rgb)
        rgb_ds = self.rgb_ds(rgb_os)

        rgb_os_s4 = self.rgb_os_s4(rgb_ds)
        rgb_ds_s4 = self.rgb_ds_s4(rgb_os_s4)

        rgb_os_s8 = self.rgb_os_s4(rgb_ds_s4)
        rgb_ds_s8 = self.rgb_ds_s4(rgb_os_s8)


        AI_ds = self.AILayer_ds(rgb_ds)
        AI = self.AILayer(rgb_os)
        AI_ds_s4 = self.AILayer_ds(rgb_ds_s4)
        AI_s4 = self.AILayer(rgb_os_s4)
        AI_ds_s8 = self.AILayer_ds(rgb_ds_s8)
        AI_s8 = self.AILayer(rgb_os_s8)

        ### SCALE 2
        f__1 = self.SFENet1(d)
        f__2 = self.SFENet2(f__1)
        x = f__2.mul(AI_ds_s8)

        RG1_out = self.resgroups[0](x)
        RG1_out += f__2
        RG1_out_up = self.UPNet(RG1_out)

        RG2_in = torch.cat([RG1_out_up, RG1_out_up.mul(AI_s8)], 1)
        RG2_in = self.fuse_conv(RG2_in)
        RG2_out = self.resgroups[1](RG2_in)
        RG2_out += RG1_out_up

        # scale 4

        RG2_out = RG2_out.mul(AI_ds_s4)
        RG1_out_s4 = self.resgroups_s4[0](RG2_out)
        RG1_out_s4 += RG2_out
        RG1_out_up_s4 = self.UPNet_s4(RG1_out_s4)

        RG2_in_s4 = torch.cat([RG1_out_up_s4, RG1_out_up_s4.mul(AI_s4)], 1)
        RG2_in_s4 = self.fuse_conv_s4(RG2_in_s4)
        RG2_out_s4 = self.resgroups_s4[1](RG2_in_s4)
        RG2_out_s4 += RG1_out_up_s4

        # scale 8

        RG2_out_s4 = RG2_out_s4.mul(AI_ds)
        RG1_out_s8 = self.resgroups_s8[0](RG2_out_s4)
        RG1_out_s8 += RG2_out_s4
        RG1_out_up_s8 = self.UPNet_s8(RG1_out_s8)

        RG2_in_s8 = torch.cat([RG1_out_up_s8, RG1_out_up_s8.mul(AI)], 1)
        RG2_in_s8 = self.fuse_conv_s8(RG2_in_s8)
        RG2_out_s8 = self.resgroups_s8[1](RG2_in_s8)
        RG2_out_s8 += RG1_out_up_s8

        out = self.tail_conv(RG2_out_s8)
        return out

    def normalizeIm(self, Im):

        Im -= np.min(Im)
        Im /= np.max(Im)


    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                # if name.find('UPNet') >= 0:
                #     print('Replace pre-trained upsampler to new one...')
                #     continue
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('UPNet') >= 0:
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