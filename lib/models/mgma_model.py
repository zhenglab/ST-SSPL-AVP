# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

import torch
from torch import nn

from lib.modules import (MGMAConvSC, MGMABasicBlock, MGMABottleneckBlock, MGMAShuffleV2Block)


class MGMA_Model(nn.Module):
    r"""MGMA Model
    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 spatio_kernel_enc=3, spatio_kernel_dec=3, act_inplace=True,
                 mgma_type='TSA', mgma_num_groups=8, block_type='i3d', middle_ratio=4, **kwargs):
        super(MGMA_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        self.hid = MidMetaNet(hid_S, hid_T, N_T,
            input_resolution=(H, W), model_type=model_type,
            mgma_type=mgma_type, mgma_num_groups=mgma_num_groups, 
            block_type=block_type, middle_ratio=middle_ratio)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for MGMA"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              MGMAConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[MGMAConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for MGMA"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[MGMAConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              MGMAConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y

class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for MGMA"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mgma_type='TSA', mgma_num_groups=8, block_type='i3d', middle_ratio=4):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'basic':
            self.block = MGMABasicBlock(in_channels, in_channels, 
                                        mgma_type=mgma_type, mgma_num_groups=mgma_num_groups, block_type=block_type)
        elif model_type == 'bottleneck':
            if in_channels < out_channels:
                inner_channels = in_channels
            else:
                inner_channels = in_channels // middle_ratio
            self.block = MGMABottleneckBlock(in_channels, in_channels, inner_channels, 
                                             mgma_type=mgma_type, mgma_num_groups=mgma_num_groups, block_type=block_type)
        elif model_type == 'shufflev2':
            self.block = MGMAShuffleV2Block(in_channels // 2, in_channels, in_channels // 2,
                                            mgma_type=mgma_type, mgma_num_groups=mgma_num_groups, block_type=block_type)
        else:
            assert False and "Invalid model_type in MGMA"

        if in_channels != out_channels:
            self.reduction = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.reduction.weight, mode='fan_out', nonlinearity='relu')
            if self.reduction.bias is not None:
                nn.init.constant_(self.reduction.bias, 0.0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for MGMA"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mgma_type='TSA', mgma_num_groups=8, 
                 block_type='i3d', middle_ratio=4):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2
        self.N2 = N2

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mgma_type=mgma_type, mgma_num_groups=mgma_num_groups, 
            block_type=block_type, middle_ratio=middle_ratio)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mgma_type=mgma_type, mgma_num_groups=mgma_num_groups, 
                block_type=block_type, middle_ratio=middle_ratio))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mgma_type=mgma_type, mgma_num_groups=mgma_num_groups, 
            block_type=block_type, middle_ratio=middle_ratio))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        # B, T, C, H, W = x.shape
        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute((0, 2, 1, 3, 4))

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        
        # (B, C, T, H, W) -> (B, T, C, H, W)
        y = z.permute((0, 2, 1, 3, 4))
        # y = z.reshape(B, T, C, H, W)
        return y
