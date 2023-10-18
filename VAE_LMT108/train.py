'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:44:36
LastEditors: Mingxin Zhang
LastEditTime: 2023-10-18 15:31:17
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import model
import time
import torch
import pickle
import os
from sklearn import preprocessing
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from scipy import stats

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'Selected device: {device}')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'trainset_LMT_large.pickle')
with open(file_path, 'rb') as file:
# with open('trainset.pickle', 'rb') as file:
    trainset = pickle.load(file)

spectrogram = torch.from_numpy(trainset['spectrogram'].astype(np.float32))
texture = trainset['texture']

# transform to [-1, 1]
def Normalization(X):
    Xmin = X.min()
    Xmax = X.max()
    X_norm = (X - Xmin) / (Xmax - Xmin)
    X_norm = 2 * X_norm - 1
    return X_norm

spectrogram = Normalization(spectrogram)

train_dataset = torch.utils.data.TensorDataset(spectrogram)
train_dataloader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = 128,
    shuffle = True,
    num_workers = 0,
    )

image_loss = nn.MSELoss()

FEAT_DIM = 256
encoder = model.ResNetEncoder(feat_dim = FEAT_DIM)
decoder = model.Decoder(feat_dim = FEAT_DIM)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()},
]

optimizer = optim.Adam(params_to_optimize, lr=5e-4, weight_decay=1e-05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5)

encoder.to(device)
decoder.to(device)

epoch_num = 100

writer = SummaryWriter()
tic = time.time()
batch_num = 0

for epoch in range(1, epoch_num + 1):
    encoder.train()
    decoder.train()

    for i, (img,) in enumerate(train_dataloader):
        batch_num += 1

        img = torch.unsqueeze(img, 1) # Add channel axis (1 channel)
        img = img.to(device)

        feat = encoder(img)
        pred_img = decoder(feat)
        loss = image_loss(pred_img, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.item(), batch_num)

    toc = time.time()

    writer.add_image('Real Spectrogram 1', img[0], epoch)
    writer.add_image('Real Spectrogram 2', img[1], epoch)
    writer.add_image('Real Spectrogram 3', img[2], epoch)
    writer.add_image('Real Spectrogram 4', img[3], epoch)
    writer.add_image('Real Spectrogram 5', img[4], epoch)
    writer.add_image('Fake Spectrogram 1', pred_img[0], epoch)
    writer.add_image('Fake Spectrogram 2', pred_img[1], epoch)
    writer.add_image('Fake Spectrogram 3', pred_img[2], epoch)
    writer.add_image('Fake Spectrogram 4', pred_img[3], epoch)
    writer.add_image('Fake Spectrogram 5', pred_img[4], epoch)

    print('=====================================================================')
    print('Epoch: ', epoch, '\tAccumulated time: ', round((toc - tic) / 3600, 4), ' hours')
    print('Loss: ', round(loss.item(), 4))
    print('=====================================================================\n')

writer.close()

torch.save(encoder.state_dict(), 'encoder_' + str(FEAT_DIM) + 'd.pt')
torch.save(decoder.state_dict(), 'decoder_' + str(FEAT_DIM) + 'd.pt')