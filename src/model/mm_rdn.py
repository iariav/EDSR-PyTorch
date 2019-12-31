# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return mm_RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3,dilation=1,padding=-1):
        super(RDB_Conv, self).__init__()

        if padding<0:
            padding = (kSize - 1) // 2
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=padding, stride=1,dilation=dilation),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))#,dilation=2**c,padding = 2**c))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args, G0,kSize,D, C, G):
        super(RDN, self).__init__()

        self.D, C, G = D, C, G

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, x):

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))

        return x


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

            self.conv_sa = nn.Conv2d(channel, channel, kernel_size, padding=(kernel_size-1)//2, groups=channel, stride=1)

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

class mm_RDN(nn.Module):
    def __init__(self, args):
        super(mm_RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        reduction = args.reduction
        conv = common.default_conv

        self.sub_mean_d = common.MeanShift(args.n_colors, args.rgb_range, rgb_mean=[0.5], rgb_std=[1.0])
        self.sub_mean_rgb = common.MeanShift(3, 255.0)
        self.add_mean_d = common.MeanShift(args.n_colors, args.rgb_range, rgb_mean=[0.5], rgb_std=[1.0], sign=1)

        # number of RDB blocks, conv layers, out channels
        D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (10, 8, 64), #  (3, 4, 32),
        }[args.RDNconfig]


        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.RDNs = nn.ModuleList()
        for i in range(2):
            self.RDNs.append(
                RDN(args,G0,kSize,D, C, G)
            )

        self.AILayer = IALayer(G, kSize, reduction, use_sa=args.use_sa, use_ca=args.use_ca)
        self.AILayer_ds = IALayer(G, kSize, reduction, use_sa=args.use_sa, use_ca=args.use_ca)

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r)
                # nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                # nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

        ### RGB BRANCH ###
        modules_rgb = [conv(3, G0, kernel_size=1),
                       conv(G0, G, kernel_size=1, padding=0, stride=1),
                       nn.ReLU(True)]

        self.rgb_os = nn.Sequential(*modules_rgb)

        modules_rgb_downsize = [conv(G, G, kernel_size=1),
                       conv(G, G, kernel_size=kSize, padding=(kSize-1)//2, stride=2),
                       nn.ReLU(True)]

        self.rgb_ds = nn.Sequential(*modules_rgb_downsize)

        self.fuse_conv = conv(G*2, G, kernel_size=1)

        self.tail_conv = conv(G, args.n_colors, kernel_size=kSize, padding=(kSize-1)//2, stride=1)

    def forward(self, d, rgb):

        # d = self.sub_mean_d(d)
        # rgb = self.sub_mean_rgb(rgb)

        rgb_os = self.rgb_os(rgb)
        rgb_ds = self.rgb_ds(rgb_os)

        AI_ds = self.AILayer_ds(rgb_ds)
        AI = self.AILayer(rgb_os)

        f__1 = self.SFENet1(d)
        x  = self.SFENet2(f__1)
        x = x.mul(AI_ds)


        RDN1_out = self.RDNs[0](x)
        RDN1_out += f__1
        RDN1_out_up = self.UPNet(RDN1_out)


        RDN2_in = torch.cat([RDN1_out_up,RDN1_out_up.mul(AI)],1)
        RDN2_in = self.fuse_conv(RDN2_in)
        RDN2_out = self.RDNs[1](RDN2_in)
        RDN2_out += RDN1_out_up

        out = self.tail_conv(RDN2_out)
        # out = self.add_mean_d(out)

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
                    elif name.find('GFF') >= 0:
                        print('Replace pre-trained GFF to new one...')
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