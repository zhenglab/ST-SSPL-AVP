# Code adapted from:
# https://github.com/zhenglab/mgma

import torch
import torch.nn as nn
import numpy as np
import copy
import math

from timm.models.layers import trunc_normal_

GN_CHANNELS = 16

class MGMA(nn.Module):
    def __init__(self, input_filters, output_filters, mgma_type="TSA", num_groups=8, freeze_bn=False, temporal_downsample=False, spatial_downsample=False):
        super(MGMA, self).__init__()
        
        self.freeze_bn = freeze_bn
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.mgma_type = mgma_type
        self.mgma_num_groups = num_groups
        
        layers = []
        if temporal_downsample and spatial_downsample:
            layers.append(self._make_downsample(kernel_type="U"))
        elif spatial_downsample:
            layers.append(self._make_downsample(kernel_type="S"))
        
        if self.input_filters % self.mgma_num_groups == 0:
            self.split_groups = [self.mgma_num_groups]
            self.split_channels = [self.input_filters]
        else:
            group_channels = math.ceil(self.input_filters / self.mgma_num_groups)
            split_groups_1 = self.input_filters // group_channels
            split_groups_2 = 1
            split_channels_1 = group_channels * split_groups_1
            split_channels_2 = self.input_filters - split_channels_1
            self.split_groups = [split_groups_1, split_groups_2]
            self.split_channels = [split_channels_1, split_channels_2]

        for i in range(len(self.split_groups)):
            if self.mgma_type in ["TA", "SA", "UA"]:
                if i == 0:
                    self.ma = nn.ModuleDict()
                layers_i = copy.deepcopy(layers)
                self._make_layers(layers_i, self.split_channels[i], self.split_channels[i], self.mgma_type[0], self.split_groups[i])
                self.ma[str(i)] = nn.Sequential(*layers_i)
            else:
                if i == 0:
                    self.ta = nn.ModuleDict()
                    self.sa = nn.ModuleDict()
                layers_t_i = copy.deepcopy(layers)
                layers_s_i = copy.deepcopy(layers)
                self._make_layers(layers_t_i, self.split_channels[i], self.split_channels[i], "T", self.split_groups[i])
                self._make_layers(layers_s_i, self.split_channels[i], self.split_channels[i], "S", self.split_groups[i])
                self.ta[str(i)] = nn.Sequential(*layers_t_i)
                self.sa[str(i)] = nn.Sequential(*layers_s_i)

    def _make_layers(self, layers, input_filters, output_filters, mgma_type, num_groups):
        layers.append(self._make_downsample(kernel_type=mgma_type))
        layers.append(self._make_interpolation(kernel_type=mgma_type))
        layers.append(self._make_conv_bn(input_filters, output_filters, kernel_type=mgma_type, groups=num_groups, use_relu=False))
        layers.append(self._make_activate())
    
    def _make_conv_bn(self, input_filters, out_filters, kernel_type="T", kernel=3, stride=1, padding=1, groups=1, use_bn=True, use_relu=True):
        layers = []
        
        if kernel_type.startswith("T"):
            layers.append(nn.Conv3d(input_filters, out_filters, kernel_size=(kernel, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), groups=groups, bias=False))
        elif kernel_type.startswith("S"):
            layers.append(nn.Conv3d(input_filters, out_filters, kernel_size=(1, kernel, kernel), stride=(1, stride, stride), padding=(0, padding, padding), groups=groups, bias=False))
        elif kernel_type.startswith("U"):    
            layers.append(nn.Conv3d(input_filters, out_filters, kernel_size=(kernel, kernel, kernel), stride=(stride, stride, stride), padding=(padding, padding, padding), groups=groups, bias=False))
        
        if use_bn:
            # layers.append(nn.BatchNorm3d(out_filters, track_running_stats=(not self.freeze_bn)))
            layers.append(nn.GroupNorm(out_filters // GN_CHANNELS, out_filters))
        if use_relu:
            # layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.SiLU(inplace=True))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def _make_downsample(self, kernel_type="T"):
        if kernel_type.startswith("T"):
            return nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        elif kernel_type.startswith("S"):
            return nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        elif kernel_type.startswith("U"):    
            return nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    
    def _make_interpolation(self, kernel_type="T"):
        if kernel_type.startswith("T"):
            self.upsample_scale_factor = (2, 1, 1)
            return nn.Upsample(scale_factor=self.upsample_scale_factor, mode='nearest')
        elif kernel_type.startswith("S"):
            self.upsample_scale_factor = (1, 2, 2)
            return nn.Upsample(scale_factor=self.upsample_scale_factor, mode='nearest')
        elif kernel_type.startswith("U"):
            self.upsample_scale_factor = (2, 2, 2)
            return nn.Upsample(scale_factor=self.upsample_scale_factor, mode='nearest')
    
    def _make_activate(self):
        return nn.Sigmoid()
        
    def forward(self, x):
        mgma_in = x
        
        if self.mgma_type in ["TA", "SA", "UA"]:
            ma_in_list = mgma_in.split(self.split_channels, 1)
            ma_out_list = []
            
            for i in range(len(ma_in_list)):
                ma_out_list.append(self.ma[str(i)](ma_in_list[i]))
            mgma_out = torch.cat(ma_out_list, 1)
            return mgma_out
        else:
            ta_in_list = mgma_in.split(self.split_channels, 1)
            ta_out_list = []
            
            for i in range(len(ta_in_list)):
                ta_out_list.append(self.ta[str(i)](ta_in_list[i]))
            mgma_ta_out = torch.cat(ta_out_list, 1)
            
            sa_in_list = mgma_in.split(self.split_channels, 1)
            sa_out_list = []
            
            for i in range(len(sa_in_list)):
                sa_out_list.append(self.sa[str(i)](sa_in_list[i]))
            mgma_sa_out = torch.cat(sa_out_list, 1)
            
            mgma_out = mgma_ta_out + mgma_sa_out
            return mgma_out

def build_conv(block, 
            in_filters, 
            out_filters, 
            kernels, 
            strides=(1, 1, 1), 
            pads=(0, 0, 0), 
            conv_idx=1, 
            block_type="3d",
            norm_module=nn.BatchNorm3d,
            freeze_bn=False):
    
    if block_type == "2.5d":
        i = 3 * in_filters * out_filters * kernels[1] * kernels[2]
        i /= in_filters * kernels[1] * kernels[2] + 3 * out_filters
        middle_filters = int(i)
        
        # 1x3x3 layer
        conv_middle = "conv{}_middle".format(str(conv_idx))
        block[conv_middle] = nn.Conv3d(
            in_filters,
            middle_filters,
            kernel_size=(1, kernels[1], kernels[2]),
            stride=(1, strides[1], strides[2]),
            padding=(0, pads[1], pads[2]),
            bias=False)
        
        bn_middle = "bn{}_middle".format(str(conv_idx))
        # block[bn_middle] = norm_module(num_features=middle_filters, track_running_stats=(not freeze_bn))
        block[bn_middle] = nn.GroupNorm(middle_filters // GN_CHANNELS, middle_filters)
        
        relu_middle = "relu{}_middle".format(str(conv_idx))
        # block[relu_middle] = nn.ReLU(inplace=True)
        # block[relu_middle] = nn.SiLU(inplace=True)
        block[relu_middle] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 3x1x1 layer
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            middle_filters,
            out_filters,
            kernel_size=(kernels[0], 1, 1),
            stride=(strides[0], 1, 1),
            padding=(pads[0], 0, 0),
            bias=False)
    elif block_type == "3d":
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            bias=False)
    elif block_type == "i3d":
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            bias=False)

def build_basic_block(block, 
                    input_filters, 
                    output_filters, 
                    use_temp_conv=False,
                    down_sampling=False, 
                    block_type='3d', 
                    norm_module=nn.BatchNorm3d,
                    freeze_bn=False):
    
    if down_sampling:
        strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)
    
    build_conv(block,
            input_filters, 
            output_filters, 
            kernels=(3, 3, 3), 
            strides=strides, 
            pads=(1, 1, 1),
            conv_idx=1,
            block_type=block_type,
            norm_module=nn.BatchNorm3d,
            freeze_bn=freeze_bn)
    # block["bn1"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    # block["relu"] = nn.ReLU(inplace=True)
    block["bn1"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)
    # block["relu"] = nn.SiLU(inplace=True)
    block["relu"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    build_conv(block,
            output_filters, 
            output_filters, 
            kernels=(3, 3, 3), 
            strides = (1, 1, 1),
            pads=(1, 1, 1), 
            conv_idx=2, 
            block_type=block_type,
            norm_module=nn.BatchNorm3d,
            freeze_bn=freeze_bn)
    # block["bn2"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    block["bn2"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)
    
    if (output_filters != input_filters) or down_sampling:
        block["shortcut"] = nn.Conv3d(
            input_filters,
            output_filters,
            kernel_size=(1, 1, 1),
            stride=strides,
            padding=(0, 0, 0),
            bias=False)
        # block["shortcut_bn"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
        block["shortcut_bn"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)

def build_bottleneck(block, 
                    input_filters, 
                    output_filters, 
                    base_filters, 
                    use_temp_conv=False,
                    down_sampling=False, 
                    block_type='3d', 
                    norm_module=nn.BatchNorm3d,
                    freeze_bn=False):
    
    if down_sampling:
        if block_type == 'i3d':
            strides = (1, 2, 2)
        else:
            strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)
    
    temp_conv_size = 3 if block_type == 'i3d' and use_temp_conv else 1
    
    build_conv(block,
            input_filters, 
            base_filters, 
            kernels=(temp_conv_size, 1, 1) if block_type == 'i3d' else (1, 1, 1), 
            pads=(temp_conv_size // 2, 0, 0) if block_type == 'i3d' else (0, 0, 0),
            conv_idx=1,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    # block["bn1"] = norm_module(num_features=base_filters, track_running_stats=(not freeze_bn))
    # block["relu1"] = nn.ReLU(inplace=True)
    block["bn1"] = nn.GroupNorm(base_filters // GN_CHANNELS, base_filters)
    # block["relu1"] = nn.SiLU(inplace=True)
    block["relu1"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    build_conv(block,
            base_filters, 
            base_filters, 
            kernels=(1, 3, 3) if block_type == 'i3d' else (3, 3, 3), 
            strides=strides,
            pads=(0, 1, 1) if block_type == 'i3d' else (1, 1, 1), 
            conv_idx=2, 
            block_type=block_type,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    # block["bn2"] = norm_module(num_features=base_filters, track_running_stats=(not freeze_bn))
    # block["relu2"] = nn.ReLU(inplace=True)
    block["bn2"] = nn.GroupNorm(base_filters // GN_CHANNELS, base_filters)
    # block["relu2"] = nn.SiLU(inplace=True)
    block["relu2"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    build_conv(block,
            base_filters, 
            output_filters, 
            kernels=(1, 1, 1), 
            conv_idx=3,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    # block["bn3"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    block["bn3"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)
    
    if (output_filters != input_filters) or down_sampling:
        block["shortcut"] = nn.Conv3d(
            input_filters,
            output_filters,
            kernel_size=(1, 1, 1),
            stride=strides,
            padding=(0, 0, 0),
            bias=False)
        # block["shortcut_bn"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
        block["shortcut_bn"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)

def init_module_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm3d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_module_weights_new(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv3d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(out_channels // GN_CHANNELS, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class MGMAConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(MGMAConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y

class MGMABasicBlock(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters,
                 mgma_type="TSA",
                 mgma_num_groups=8,
                 use_temp_conv=False,
                 down_sampling=False, 
                 block_type='3d', 
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(MGMABasicBlock, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.mgma_type = mgma_type
        self.down_sampling = down_sampling
        self.block = nn.ModuleDict()
        
        build_basic_block(self.block, 
                          input_filters, 
                          output_filters, 
                          use_temp_conv,
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if self.mgma_type != 'NONE':
            self.mgma = MGMA(input_filters, output_filters, mgma_type, mgma_num_groups, freeze_bn=freeze_bn, 
                             temporal_downsample=down_sampling, spatial_downsample=down_sampling)
        
        for m in self.modules():
            init_module_weights_new(m)
    
    def channel_shuffle(self, x):
        batchsize, num_channels, temporal, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, temporal * height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, temporal, height, width)
        return torch.cat((x[0], x[1]), 1) 
        
    def forward(self, x):
        if self.mgma_type != 'NONE':
            x = self.channel_shuffle(x)
            mgma_out = x
        
        residual = x
        out = x
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        out += residual
        out = self.block["relu"](out)
        
        if self.mgma_type != 'NONE':
            mgma_out = self.mgma(mgma_out)
            mgma_out = out * mgma_out
            out = out + mgma_out

        return out

class MGMABottleneckBlock(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters, 
                 base_filters, 
                 mgma_type="TSA",
                 mgma_num_groups=8,
                 use_temp_conv=True,
                 down_sampling=False, 
                 block_type='i3d', 
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(MGMABottleneckBlock, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.base_filters = base_filters
        self.mgma_type = mgma_type
        self.down_sampling = down_sampling
        self.block = nn.ModuleDict()
        
        build_bottleneck(self.block, 
                          input_filters, 
                          output_filters, 
                          base_filters, 
                          use_temp_conv, 
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if self.mgma_type != 'NONE':
            self.mgma = MGMA(input_filters, output_filters, mgma_type, mgma_num_groups, freeze_bn=freeze_bn, 
                             temporal_downsample=down_sampling, spatial_downsample=down_sampling)
        
        for m in self.modules():
            init_module_weights_new(m)
    
    def channel_shuffle(self, x):
        batchsize, num_channels, temporal, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, temporal * height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, temporal, height, width)
        return torch.cat((x[0], x[1]), 1) 
        
    def forward(self, x):
        if self.mgma_type != 'NONE':
            x = self.channel_shuffle(x)
            mgma_out = x
        
        residual = x
        out = x
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        out += residual
        out = self.block["relu1"](out)
        
        if self.mgma_type != 'NONE':
            mgma_out = self.mgma(mgma_out)
            mgma_out = out * mgma_out
            out = out + mgma_out

        return out

class MGMAShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize=3, stride=1, temporal_downsample=False, block_type="i3d", mgma_type="TSA",
                mgma_num_groups=8, freeze_bn=False, norm_module=nn.BatchNorm3d):
        super(MGMAShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp
        
        self.mgma_type = mgma_type

        if block_type == "i3d":
            branch_main = [
                # pw
                nn.Conv3d(inp, mid_channels, kernel_size=(3, 1, 1), stride=(2 if temporal_downsample else 1, 1, 1), padding=(1, 0, 0), bias=False),
                # norm_module(mid_channels),
                # nn.ReLU(inplace=True),
                nn.GroupNorm(mid_channels // GN_CHANNELS, mid_channels),
                # nn.SiLU(inplace=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # dw
                nn.Conv3d(mid_channels, mid_channels, kernel_size=(1, ksize, ksize), stride=(1, stride, stride), padding=(0, pad, pad), groups=mid_channels, bias=False),
                # norm_module(mid_channels),
                nn.GroupNorm(mid_channels // GN_CHANNELS, mid_channels),
                # pw-linear
                nn.Conv3d(mid_channels, outputs, 1, 1, 0, bias=False),
                # norm_module(outputs),
                # nn.ReLU(inplace=True),
                nn.GroupNorm(outputs // GN_CHANNELS, outputs),
                # nn.SiLU(inplace=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
        else:
            branch_main = [
                # pw
                nn.Conv3d(inp, mid_channels, 1, 1, 0, bias=False),
                # norm_module(mid_channels),
                # nn.ReLU(inplace=True),
                nn.GroupNorm(mid_channels // GN_CHANNELS, mid_channels),
                # nn.SiLU(inplace=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                # dw
                nn.Conv3d(mid_channels, mid_channels, kernel_size=(3, ksize, ksize), stride=(2 if temporal_downsample else 1, stride, stride), padding=(1, pad, pad), groups=mid_channels, bias=False),
                # norm_module(mid_channels),
                nn.GroupNorm(mid_channels // GN_CHANNELS, mid_channels),
                # pw-linear
                nn.Conv3d(mid_channels, outputs, 1, 1, 0, bias=False),
                # norm_module(outputs),
                # nn.ReLU(inplace=True),
                nn.GroupNorm(outputs // GN_CHANNELS, outputs),
                # nn.SiLU(inplace=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            if block_type == "i3d":
                branch_proj = [
                    # dw
                    nn.Conv3d(inp, inp, kernel_size=(1, ksize, ksize), stride=(1, stride, stride), padding=(0, pad, pad), groups=inp, bias=False),
                    # norm_module(inp),
                    nn.GroupNorm(inp // GN_CHANNELS, inp),
                    # pw-linear
                    nn.Conv3d(inp, inp, kernel_size=(3, 1, 1), stride=(2 if temporal_downsample else 1, 1, 1), padding=(1, 0, 0), bias=False),
                    # norm_module(inp),
                    # nn.ReLU(inplace=True),
                    nn.GroupNorm(inp // GN_CHANNELS, inp),
                    # nn.SiLU(inplace=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            else:
                branch_proj = [
                    # dw
                    nn.Conv3d(inp, inp, kernel_size=(3, ksize, ksize), stride=(2 if temporal_downsample else 1, stride, stride), padding=(1, pad, pad), groups=inp, bias=False),
                    # norm_module(inp),
                    nn.GroupNorm(inp // GN_CHANNELS, inp),
                    # pw-linear
                    nn.Conv3d(inp, inp, 1, 1, 0, bias=False),
                    # norm_module(inp),
                    # nn.ReLU(inplace=True),
                    nn.GroupNorm(inp // GN_CHANNELS, inp),
                    # nn.SiLU(inplace=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

        if self.mgma_type != 'NONE':
            input_filters = inp * 2 if self.stride == 1 else inp
            output_filters = oup
            self.mgma = MGMA(input_filters, output_filters, mgma_type, mgma_num_groups, freeze_bn=freeze_bn, 
                             temporal_downsample=temporal_downsample, spatial_downsample=(stride == 2))
        
        for m in self.modules():
            init_module_weights_new(m)

    def forward(self, old_x):
        # shuffle
        if self.stride==1:
            if self.mgma_type != 'NONE':
                x_proj, x = self.channel_shuffle(old_x)
                x_mgma_in = torch.cat((x_proj, x), 1)
                x_block_out = torch.cat((x_proj, self.branch_main(x)), 1)
            else:
                x_proj, x = self.channel_shuffle(old_x)
                x_block_out = torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            if self.mgma_type != 'NONE':
                x_mgma_in = old_x
            x_block_out = torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        
        # output    
        if self.mgma_type != 'NONE':
            x_mgma_out = self.mgma(x_mgma_in)
            x_mgma_out = x_block_out * x_mgma_out
            x_block_out = x_block_out + x_mgma_out
            return x_block_out
        else:
            return x_block_out

    def channel_shuffle(self, x):
        batchsize, num_channels, temporal, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, temporal * height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, temporal, height, width)
        return x[0], x[1]

