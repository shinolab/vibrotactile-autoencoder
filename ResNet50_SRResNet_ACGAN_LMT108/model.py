'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:41:24
LastEditors: Mingxin Zhang
LastEditTime: 2023-09-27 01:37:55
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import math
import torch
import torchvision
import torch.nn.functional as F
from torch import nn


class ResNetEncoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(ResNetEncoder, self).__init__()

        self.flatten = nn.Flatten(start_dim=1)
        self.resize_input = nn.Linear(12 * 160, 3 * 128 * 128)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(3, 128, 128))

        self.res50 = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        numFit = self.res50.fc.in_features
        self.res50.fc = nn.Linear(numFit, encoded_space_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.resize_input(x)
        x = self.unflatten(x)

        x = self.res50(x)
        return x


class LatentDiscriminator(nn.Module):
    def __init__(self, encoded_space_dim):
        super(LatentDiscriminator, self).__init__()

        self.flatten = nn.Flatten(start_dim=1)
        self.resize = nn.Linear(encoded_space_dim, 16 * 8)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 16, 8))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(128 * 512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_d = nn.Linear(256, 1)

    def forward(self, x):
        x = self.unflatten(self.resize(x))
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.flatten(x)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)

        out_d = self.fc_d(x)
        out_d = F.sigmoid(out_d)
        return out_d


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output


class Generator(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Generator, self).__init__()

        self.resize = nn.Linear(encoded_space_dim, 3 * 40)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 3, 40))

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.unflatten(self.resize(x))
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = F.tanh(self.conv_output(out))
        return out


class SpectrogramDiscriminator(nn.Module):
    def __init__(self):
        super(SpectrogramDiscriminator, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 2 * 11, 1024)
        self.fc_d = nn.Linear(1024, 1)
        self.fc_c = nn.Linear(1024, 7)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.LeakyReLU(out)

        out_d = self.fc_d(out)
        out_d = self.sigmoid(out_d)

        out_c = self.fc_c(out)
        out_c = self.softmax(out_c)
        return out_d.view(-1, 1), out_c