import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def act(act_fun='LeakyReLU'):
    """
        Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        # elif act_fun == 'Swish':
        #     return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()

def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        # elif downsample_mode in ['lanczos2', 'lanczos3']:
        #     downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5,
        #                               preserve_size=True)
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = [x for x in [padder, convolver, downsampler] if x is not None]
    return nn.Sequential(*layers)


class down_block(nn.Module):
    def __init__(self, in_f, out_f, kernel_size, bias=True, pad='zero', downsample_mode='stride', act_fun='LeakyReLU'):
        super(down_block, self).__init__()
        self.conv1 = conv(in_f, out_f, kernel_size, 2, bias, pad, downsample_mode)
        self.bn1 = bn(out_f)
        self.act1 = act(act_fun)
        self.conv2 = conv(out_f, out_f, kernel_size, 1, bias, pad, downsample_mode)
        self.bn2 = bn(out_f)
        self.act2 = act(act_fun)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        return out

class up_block(nn.Module):
    def __init__(self, in_f, out_f, kernel_size, bias=True, pad='zero', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
        super(up_block, self).__init__()
        self.need1x1_up = need1x1_up
        self.bn0 = bn(in_f)
        self.conv1 = conv(in_f, out_f, kernel_size, 1, bias, pad, downsample_mode)
        self.bn1 = bn(out_f)
        self.act1 = act(act_fun)
        self.conv2 = conv(out_f, out_f, 1, 1, bias, pad, downsample_mode)
        self.bn2 = bn(out_f)
        self.act2 = act(act_fun)

    def forward(self, x):
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act1(out)
        if self.need1x1_up:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.act2(out)

        return out

class dip_segment(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, num_channels_down=[8, 16, 32],
        num_channels_up=[8, 16, 32],
        num_channels_skip=[0, 0, 0], filter_size_down=3,
        filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):

        super(dip_segment, self).__init__()
        self.need_sigmoid = need_sigmoid
        self.upsample_mode = upsample_mode
        self.down_block1 = down_block(num_input_channels, num_channels_down[0], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.down_block2 = down_block(num_channels_down[0], num_channels_down[1], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.down_block3 = down_block(num_channels_down[1], num_channels_down[2], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.up_block3 = up_block(num_channels_down[2], num_channels_up[2], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.up_block2 = up_block(num_channels_up[2], num_channels_up[1], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.up_block1 = up_block(num_channels_up[1], num_channels_up[0], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)

        self.predict = conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        shape0 = x.shape[2:]
        out = self.down_block1(x)
        shape1 = out.shape[2:]

        out = self.down_block2(out)
        shape2 = out.shape[2:]

        out = self.down_block3(out)
        shape3 = out.shape[2:]

        out = F.upsample(out, size=shape2, mode=self.upsample_mode, align_corners=True)
        out = self.up_block3(out)

        out = F.upsample(out, size=shape1, mode=self.upsample_mode, align_corners=True)
        out = self.up_block2(out)

        out = F.upsample(out, size=shape0, mode=self.upsample_mode, align_corners=True)
        out = self.up_block1(out)

        out = self.predict(out)
        if self.need_sigmoid:
            out = self.sigm(out)

        return out


class dip_segment_deeper(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, num_channels_down=[8, 16, 32, 64],
        num_channels_up=[8, 16, 32, 64],
        num_channels_skip=[0, 0, 0], filter_size_down=3,
        filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):

        super(dip_segment_deeper, self).__init__()
        self.need_sigmoid = need_sigmoid
        self.upsample_mode = upsample_mode
        self.down_block1 = down_block(num_input_channels, num_channels_down[0], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.down_block2 = down_block(num_channels_down[0], num_channels_down[1], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.down_block3 = down_block(num_channels_down[1], num_channels_down[2], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.down_block4 = down_block(num_channels_down[2], num_channels_down[3], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.up_block4 = up_block(num_channels_down[3], num_channels_up[3], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.up_block3 = up_block(num_channels_down[3], num_channels_up[2], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.up_block2 = up_block(num_channels_up[2], num_channels_up[1], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.up_block1 = up_block(num_channels_up[1], num_channels_up[0], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)

        self.predict = conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        shape0 = x.shape[2:]
        out = self.down_block1(x)
        shape1 = out.shape[2:]

        out = self.down_block2(out)
        shape2 = out.shape[2:]

        out = self.down_block3(out)
        shape3 = out.shape[2:]

        out = self.down_block4(out)

        out = F.upsample(out, size=shape3, mode=self.upsample_mode, align_corners=True)
        out = self.up_block4(out)

        out = F.upsample(out, size=shape2, mode=self.upsample_mode, align_corners=True)
        out = self.up_block3(out)

        out = F.upsample(out, size=shape1, mode=self.upsample_mode, align_corners=True)
        out = self.up_block2(out)

        out = F.upsample(out, size=shape0, mode=self.upsample_mode, align_corners=True)
        out = self.up_block1(out)

        out = self.predict(out)
        if self.need_sigmoid:
            out = self.sigm(out)

        return out


class dip_segment_dropout(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, num_channels_down=[8, 16, 32],
        num_channels_up=[8, 16, 32],
        num_channels_skip=[0, 0, 0], filter_size_down=3,
        filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):

        super(dip_segment_dropout, self).__init__()
        self.need_sigmoid = need_sigmoid
        self.upsample_mode = upsample_mode
        self.down_block1 = down_block(num_input_channels, num_channels_down[0], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.down_block2 = down_block(num_channels_down[0], num_channels_down[1], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.down_block3 = down_block(num_channels_down[1], num_channels_down[2], filter_size_down, bias=need_bias,
                                      pad=pad, downsample_mode=downsample_mode, act_fun=act_fun)
        self.up_block3 = up_block(num_channels_down[2], num_channels_up[2], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.up_block2 = up_block(num_channels_up[2], num_channels_up[1], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)
        self.up_block1 = up_block(num_channels_up[1], num_channels_up[0], filter_size_up, bias=need_bias,
                                  pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)

        self.predict = conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        shape0 = x.shape[2:]
        out = self.down_block1(x)
        shape1 = out.shape[2:]

        out = self.down_block2(out)
        shape2 = out.shape[2:]

        out = self.down_block3(out)
        shape3 = out.shape[2:]

        out = F.upsample(out, size=shape2, mode=self.upsample_mode, align_corners=True)
        out = F.dropout2d(out, 0.7)
        out = self.up_block3(out)

        out = F.upsample(out, size=shape1, mode=self.upsample_mode, align_corners=True)
        out = F.dropout2d(out, 0.7)
        out = self.up_block2(out)

        out = F.upsample(out, size=shape0, mode=self.upsample_mode, align_corners=True)
        out = F.dropout2d(out, 0.7)
        out = self.up_block1(out)

        out = self.predict(out)
        if self.need_sigmoid:
            out = self.sigm(out)

        return out