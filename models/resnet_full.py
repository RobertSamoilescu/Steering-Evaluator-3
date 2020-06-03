# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from collections import OrderedDict

from pipeline.utils import *


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(Decoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.convs[("dispconv", i)](x)

        return self.outputs


class RESNET(nn.Module):
    def __init__(self, no_outputs, use_rgb=False, use_stacked=False, use_disp=False, 
            use_depth=False, use_flow=False, use_speed=False):
        super(RESNET, self).__init__()
        self.no_outputs = no_outputs
        self.use_rgb = use_rgb
        self.use_stacked = use_stacked
        self.use_disp = use_disp
        self.use_depth = use_depth
        self.use_flow = use_flow
        self.use_speed = use_speed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_channels = 0
        if self.use_rgb:
            self.input_channels += 6 if self.use_stacked else 3
        if self.use_depth or self.use_disp:
            self.input_channels += 1
            self.depth_net = Monodepth()
        if self.use_flow:
            self.input_channels += 2
            self.flow_net = Flow()

        # define encoder
        self.encoder = ResnetEncoder(num_layers=18, pretrained=True, num_input_images=1)
        self.encoder.encoder.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 8), stride=4, padding=0)
        self.classifier = nn.Linear(512 * 1 * 1 + (1 if self.use_speed else 0), self.no_outputs)

    def forward(self, data):
        B, _, H, W = data["img"].shape
        input = []

        mean_rgb = torch.tensor([0.47, 0.44, 0.45]).view(1, 3, 1, 1).to(self.device)
        std_rgb = torch.tensor([0.22, 0.22, 0.22]).view(1, 3, 1, 1).to(self.device)

        mean_depth = torch.tensor([20.38]).view(1, 1, 1, 1).to(self.device)
        std_depth = torch.tensor([16.53]).view(1, 1, 1, 1).to(self.device)

        mean_disp = torch.tensor([0.21]).view(1, 1, 1, 1).to(self.device)
        std_disp = torch.tensor([0.16]).view(1, 1, 1, 1).to(self.device)
        
        mean_flow = torch.tensor([-2.80, -2.74]).view(1, 2, 1, 1).to(self.device)
        std_flow = torch.tensor([20.01, 10.38]).view(1, 2, 1, 1).to(self.device)

        if self.use_rgb:
            img = data["img"]
            img = (img - mean_rgb) / std_rgb
            input.append(img)

            if self.use_stacked:
                prev_img = data["prev_img"]
                prev_img = (prev_img - mean_rgb) / std_rgb
                input.append(prev_img)


        orig_disp, disp = None, None
        if self.use_disp:
            img1 = F.interpolate(data["img"], (256, 512))
            disp = self.depth_net.forward(img1).to(self.device)
            orig_disp = disp.clone()
            disp = (disp - mean_disp) / std_disp
            disp = F.interpolate(disp, (H, W))
            input.append(disp)

        orig_depth, depth = None, None
        if self.use_depth:
            img1 = F.interpolate(data["img"], (256, 512))
            tmp_disp = self.depth_net.forward(img1)
            depth = self.depth_net.get_depth(tmp_disp).to(self.device)
            orig_depth = depth.clone()
            depth = (depth - mean_depth) / std_depth
            depth = F.interpolate(depth, (H, W))
            input.append(depth)

        orig_flow, flow = None, None
        if self.use_flow:
            img1 = F.interpolate(data["prev_img"], (256, 512))
            img2 = F.interpolate(data["img"], (256, 512))
            flow = self.flow_net.get_flow(img1, img2, B).to(self.device)
            orig_flow = flow.clone()
            flow = (flow - mean_flow) / std_flow
            flow = F.interpolate(flow, (H, W))
            input.append(flow)
            
        input = torch.cat(input, dim=1)
        input = self.encoder(input)[4]
        input = self.avg_pool(input)
        input = input.reshape(input.shape[0], -1)
        if self.use_speed:
            input = torch.cat([input, data["speed"]], dim=1)
        output = self.classifier(input)
        return output, orig_disp, orig_depth, orig_flow

