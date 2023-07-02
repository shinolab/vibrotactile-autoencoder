'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:44:36
LastEditors: Mingxin Zhang
LastEditTime: 2023-07-01 13:42:28
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'trainset_LMT_large.pickle')
with open(file_path, 'rb') as file:
# with open('trainset.pickle', 'rb') as file:
    trainset = pickle.load(file)

spectrogram = torch.from_numpy(trainset['spectrogram'].astype(np.float32))
texture = trainset['texture']
le = preprocessing.LabelEncoder()
onehot = preprocessing.OneHotEncoder()
labels = le.fit_transform(texture)
labels = torch.as_tensor(onehot.fit_transform(labels.reshape(-1, 1)).toarray())

# transform to [-1, 1]
def Normalization(X):
    Xmin = X.min()
    Xmax = X.max()
    X_norm = (X - Xmin) / (Xmax - Xmin)
    X_norm = 2 * X_norm - 1
    return X_norm

spectrogram = Normalization(spectrogram)

train_dataset = torch.utils.data.TensorDataset(spectrogram, labels)
train_dataloader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True,
    num_workers = 0,
    )

adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

FEAT_DIM = 256
CLASS_NUM = 108

generator= model.Generator(encoded_space_dim = FEAT_DIM)
dis_spec = model.SpectrogramDiscriminator()

gen_lr = 1e-4
d_spec_lr = 1e-4

optimizer_G = optim.Adam(generator.parameters(), lr=gen_lr)
optimizer_D_spec = optim.Adam(dis_spec.parameters(), lr=d_spec_lr)

generator.to(device)
dis_spec.to(device)

epoch_num = 150

writer = SummaryWriter()
tic = time.time()
batch_num = 0

for epoch in range(1, epoch_num + 1):
    generator.train()
    dis_spec.train()

    for i, (img, label) in enumerate(train_dataloader):
        batch_num += 1

        img = torch.unsqueeze(img, 1) # Add channel axis (1 channel)
        img = img.to(device)
        label = label.to(device)

        soft_scale = 0.1
        valid = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        soft_valid = valid - torch.rand(img.size(0), 1).to(device) * soft_scale
        fake = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(0.0), requires_grad=False).to(device)
        soft_fake = fake + torch.rand(img.size(0), 1).to(device) * soft_scale

        # 1) reconstruction
        # 1.1) generator
        for i in range(5):
            optimizer_G.zero_grad()
            # input latent vector
            z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (img.shape[0], FEAT_DIM - CLASS_NUM)))).to(device)
            z = torch.cat((z, label), dim=1)
            # train generator
            gen_img = generator(z)
            output_d, output_c = dis_spec(gen_img)

            g_loss = (adversarial_loss(output_d, valid) + auxiliary_loss(output_c, label)) / 2
            g_loss.backward()
            optimizer_G.step()

        writer.add_scalar('Spectrogram/G_loss', g_loss.item(), batch_num)

        # 1.2) spectrogram discriminator
        optimizer_D_spec.zero_grad()
        # loss for real img
        output_d, output_c = dis_spec(img)
        real_loss = (adversarial_loss(output_d, soft_valid) + auxiliary_loss(output_c, label)) / 2

        # loss for fake img
        z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (img.shape[0], FEAT_DIM - CLASS_NUM)))).to(device)
        z = torch.cat((z, label), dim=1)
        gen_img = generator(z)
        output_d, output_c = dis_spec(gen_img.detach())
        fake_loss = (adversarial_loss(output_d, soft_fake) + auxiliary_loss(output_c, label)) / 2

        d_spec_loss = (real_loss + fake_loss) / 2

        d_spec_loss.backward()
        optimizer_D_spec.step()

        writer.add_scalar('Spectrogram/D_loss', d_spec_loss.item(), batch_num)

    toc = time.time()

    writer.add_image('Real Spectrogram', img[0], epoch)
    writer.add_image('Fake Spectrogram', gen_img[0], epoch)

    print('=====================================================================')
    print('Epoch: ', epoch, '\tAccumulated time: ', round((toc - tic) / 3600, 4), ' hours')
    print('Generator Loss: ', round(g_loss.item(), 4), '\tSpec Discriminator Loss: ', round(d_spec_loss.item(), 4))
    print('=====================================================================\n')

writer.close()

torch.save(generator.state_dict(), 'generator_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_spec.state_dict(), 'dis_spec_' + str(FEAT_DIM) + 'd.pt')