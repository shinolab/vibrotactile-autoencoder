'''
Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
Date: 2023-04-12 01:41:18
LastEditors: Mingxin Zhang
LastEditTime: 2023-04-12 14:58:18
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

from torch import nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(32 * 1 * 23, 128)
        self.fc2 = nn.Linear(128, encoded_space_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, i_1 = self.pool1(self.conv2(x))
        x = F.relu(self.bn1(x))
        x, i_2 = self.pool2(self.conv3(x))
        x = F.relu(self.bn2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, i_1, i_2


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.fc1 = nn.Linear(encoded_space_dim, 128)
        self.fc2 = nn.Linear(128, 32 * 1 * 23)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 1, 23))

        self.deconv1 = nn.ConvTranspose2d(32, 16, 5, output_padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 3, padding=1)
        
    def forward(self, x, unpool_i1, unpool_i2):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x)
        x = self.unpool1(x, unpool_i2)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.unpool2(x, unpool_i1)
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        return x


class Classifier(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.fc1 = nn.Linear(encoded_space_dim, 16)
        self.fc2 = nn.Linear(16, 7)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1) #dim0=batch, dim1=element
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim)
        self.classifier = Classifier(encoded_space_dim)
        
    def forward(self, x):
        feat, i_1, i_2 = self.encoder(x)
        pred_img = self.decoder(feat, i_1, i_2)
        pred_label = self.classifier(feat)
        return feat, pred_img, pred_label
    

# Used for test without indices of pooling
class TestDecoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.fc1 = nn.Linear(encoded_space_dim, 128)
        self.fc2 = nn.Linear(128, 32 * 1 * 23)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 1, 23))

        self.deconv1 = nn.ConvTranspose2d(32, 16, 5, output_padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 3, padding=1)
    
    def upsample(self, A):
        size = torch.tensor(A.shape)
        size[-2:] *= 2
        B = torch.zeros(list(size))
        B[..., ::2, ::2] = A
        # return B.to("mps")
        return B.to("cpu")
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x)
        x = self.upsample(x)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        return x