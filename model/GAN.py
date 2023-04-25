'''
Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
Date: 2023-04-12 01:41:18
LastEditors: Mingxin Zhang
LastEditTime: 2023-04-25 17:53:55
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

from torch import nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.fc1 = nn.Linear(encoded_space_dim, 32 * 1 * 23)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 1, 23))

        self.deconv1 = nn.ConvTranspose2d(32, 16, 5, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.unflatten(x)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        ### Convolutional section
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(32 * 1 * 23, 1)

        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pool(self.conv2(x))
        x = self.dropout(F.leaky_relu(self.bn1(x), 0.2))
        x = self.pool(self.conv3(x))
        x = self.dropout(F.leaky_relu(self.bn2(x), 0.2))
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        return x