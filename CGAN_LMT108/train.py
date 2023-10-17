'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:44:36
LastEditors: Mingxin Zhang
LastEditTime: 2023-10-18 03:27:14
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
labels = torch.as_tensor(le.fit_transform(texture))

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
    batch_size = 128,
    shuffle = True,
    num_workers = 0,
    )

adversarial_loss = nn.BCELoss()

FEAT_DIM = 256
CLASS_NUM = 108
generator= model.Generator(feat_dim = FEAT_DIM, class_dim = CLASS_NUM)
dis_spec = model.SpectrogramDiscriminator(class_dim = CLASS_NUM)

gen_lr = 1e-4
d_spec_lr = 1e-4

optimizer_G = optim.Adam(generator.parameters(), lr=gen_lr)
optimizer_D_spec = optim.Adam(dis_spec.parameters(), lr=d_spec_lr)

generator.to(device)
dis_spec.to(device)

epoch_num = 50

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

        soft_scale = 0.2
        valid = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        fake = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(0.0), requires_grad=False).to(device)

        # 1) reconstruction
        # 1.1) generator
        optimizer_G.zero_grad()
        # input latent vector
        z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (img.shape[0], FEAT_DIM)))).to(device)
        # train generator
        fake_label = torch.LongTensor(np.random.randint(0, CLASS_NUM, img.size(0))).to(device)
        gen_img = generator(z, fake_label)
        output_d = dis_spec(gen_img, fake_label)

        g_loss = adversarial_loss(output_d, valid)
        g_loss.backward()
        optimizer_G.step()

        writer.add_scalar('Spectrogram/G_loss', g_loss.item(), batch_num)

        # 1.2) spectrogram discriminator
        soft_valid = valid - torch.rand(img.size(0), 1).to(device) * soft_scale
        soft_fake = fake + torch.rand(img.size(0), 1).to(device) * soft_scale

        optimizer_D_spec.zero_grad()
        z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (img.shape[0], FEAT_DIM)))).to(device)
        gen_img = generator(z, fake_label)
        
        # loss for real img
        output_d = dis_spec(img, label)
        real_loss = adversarial_loss(output_d, soft_valid)

        # loss for fake img
        fake_label = torch.LongTensor(np.random.randint(0, CLASS_NUM, img.size(0))).to(device)
        output_d = dis_spec(gen_img.detach(), fake_label)
        fake_loss = adversarial_loss(output_d, soft_fake)

        d_spec_loss = (real_loss + fake_loss) / 2

        d_spec_loss.backward()
        optimizer_D_spec.step()

        writer.add_scalar('Spectrogram/D_loss', d_spec_loss.item(), batch_num)

    toc = time.time()

    writer.add_image('Real Spectrogram 1', img[0], epoch)
    writer.add_image('Real Spectrogram 2', img[1], epoch)
    writer.add_image('Real Spectrogram 3', img[2], epoch)
    writer.add_image('Real Spectrogram 4', img[3], epoch)
    writer.add_image('Real Spectrogram 5', img[4], epoch)
    writer.add_image('Fake Spectrogram 1', gen_img[0], epoch)
    writer.add_image('Fake Spectrogram 2', gen_img[1], epoch)
    writer.add_image('Fake Spectrogram 3', gen_img[2], epoch)
    writer.add_image('Fake Spectrogram 4', gen_img[3], epoch)
    writer.add_image('Fake Spectrogram 5', gen_img[4], epoch)

    print('=====================================================================')
    print('Epoch: ', epoch, '\tAccumulated time: ', round((toc - tic) / 3600, 4), ' hours')
    print('Generator Loss: ', round(g_loss.item(), 4), '\tSpec Discriminator Loss: ', round(d_spec_loss.item(), 4))
    print('=====================================================================\n')

writer.close()

torch.save(generator.state_dict(), 'generator_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_spec.state_dict(), 'dis_spec_' + str(FEAT_DIM) + 'd.pt')