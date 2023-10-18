'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:41:24
LastEditors: Mingxin Zhang
LastEditTime: 2023-10-18 15:33:52
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import math
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torch import nn


class ResNetEncoder(nn.Module):
    def __init__(self, feat_dim):
        super(ResNetEncoder, self).__init__()

        self.flatten = nn.Flatten(start_dim=1)

        self.resize_x = nn.Linear(48 * 320, 3 * 128 * 128)
        self.unflatten_x = nn.Unflatten(dim=1, unflattened_size=(3, 128, 128))

        self.res50 = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        numFit = self.res50.fc.in_features
        self.res50.fc = nn.Linear(numFit, 1024)

        self.fc1 = nn.Linear(1024, feat_dim)
        self.fc2 = nn.Linear(1024, feat_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(mu)
        z = mu + std * epsilon
        return z

    def forward(self, x):
        x = self.flatten(x)
        x = self.resize_x(x)
        x = self.unflatten_x(x)
        x = self.res50(x)

        mu = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mu, logvar)
        return z


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


class Decoder(nn.Module):
    def __init__(self, feat_dim):
        super(Decoder, self).__init__()
        self.feat_dim = feat_dim

        self.resize_x = nn.Linear(feat_dim, 12 * 80)
        self.unflatten_x = nn.Unflatten(dim=1, unflattened_size=(1, 12, 80))

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
        x = self.unflatten_x(self.resize_x(x))
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = F.tanh(self.conv_output(out))
        return out
    
    def calc_model_gradient(self, latent_vector, device):
        jacobian = self.calc_model_gradient_FDM(latent_vector, device, delta=1e-2)
        return jacobian

    def calc_model_gradient_FDM(self, latent_vector, device, delta=1e-4):
        sample_latents = np.repeat(latent_vector.reshape(1, -1).cpu(), repeats=self.feat_dim + 1, axis=0)
        sample_latents[1:] += np.identity(self.feat_dim) * delta

        sample_datas = self.forward(sample_latents.to(device))
        sample_datas = sample_datas.reshape(-1, 48*320)

        jacobian = (sample_datas[1:] - sample_datas[0]).T / delta
        return jacobian

