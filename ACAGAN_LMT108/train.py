'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:44:36
LastEditors: Mingxin Zhang
LastEditTime: 2023-10-16 13:08:44
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
image_loss = nn.MSELoss()

FEAT_DIM = 256
encoder = model.ResNetEncoder(encoded_space_dim = FEAT_DIM)
generator= model.Generator(encoded_space_dim = FEAT_DIM)
dis_latent = model.LatentDiscriminator(encoded_space_dim = FEAT_DIM)
dis_spec = model.SpectrogramDiscriminator()

gen_lr = 1e-4
encoder_lr = 1e-4
d_spec_lr = 1e-4
d_latent_lr = 1e-4

optimizer_G = optim.Adam(generator.parameters(), lr=gen_lr)
optimizer_E = optim.Adam(encoder.parameters(), lr=encoder_lr)
optimizer_D_spec = optim.Adam(dis_spec.parameters(), lr=d_spec_lr)
optimizer_D_latent = optim.Adam(dis_latent.parameters(), lr=d_latent_lr)

encoder.to(device)
dis_latent.to(device)
generator.to(device)
dis_spec.to(device)

epoch_num = 100

writer = SummaryWriter()
tic = time.time()
batch_num = 0

for epoch in range(1, epoch_num + 1):
    encoder.train()
    dis_latent.train()
    generator.train()
    dis_spec.train()

    for i, (img, label) in enumerate(train_dataloader):
        batch_num += 1

        img = torch.unsqueeze(img, 1) # Add channel axis (1 channel)
        img = img.to(device)
        label = label.to(device)

        soft_scale = 0.3
        valid = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        soft_valid = valid - torch.rand(img.size(0), 1).to(device) * soft_scale
        fake = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(0.0), requires_grad=False).to(device)
        soft_fake = fake + torch.rand(img.size(0), 1).to(device) * soft_scale

        # 1) reconstruction
        # 1.1) generator
        for i in range(5):
            optimizer_G.zero_grad()
            # input latent vector
            z = encoder(img)
            # train generator
            gen_img = generator(z)
            output_d = dis_spec(gen_img)

            g_loss = adversarial_loss(output_d, valid) + 50 * image_loss(gen_img, img)
            g_loss.backward()
            optimizer_G.step()

        writer.add_scalar('Spectrogram/G_loss', g_loss.item(), batch_num)

        # 1.2) spectrogram discriminator
        optimizer_D_spec.zero_grad()
        z = encoder(img)
        gen_img = generator(z)
        
        # loss for real img
        output_d = dis_spec(img)
        real_loss = adversarial_loss(output_d, soft_valid)

        # loss for fake img
        output_d = dis_spec(gen_img.detach())
        fake_loss = adversarial_loss(output_d, soft_fake)

        d_spec_loss = (real_loss + fake_loss) / 2

        d_spec_loss.backward()
        optimizer_D_spec.step()

        writer.add_scalar('Spectrogram/D_loss', d_spec_loss.item(), batch_num)

        # 2) latent discriminator
        for i in range(5):
            optimizer_D_latent.zero_grad()
            real_z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (img.shape[0], FEAT_DIM)))).to(device)
            fake_z = encoder(img)

            # loss for real distribution
            output_d = dis_latent(real_z)
            real_loss = adversarial_loss(output_d, soft_valid)
            # loss for fake distribution
            output_d = dis_latent(fake_z.detach())
            fake_loss = adversarial_loss(output_d, soft_fake)

            d_latent_loss = (real_loss + fake_loss) / 2
            d_latent_loss.backward()
            optimizer_D_latent.step()

        writer.add_scalar('Latent/D_loss', d_latent_loss.item(), batch_num)

        # 3) encoder
        optimizer_E.zero_grad()
        fake_z = encoder(img)

        output_d = dis_latent(fake_z)

        E_loss = adversarial_loss(output_d, valid)
        E_loss.backward()
        optimizer_E.step()

        writer.add_scalar('Latent/G_loss', E_loss.item(), batch_num)

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
    print('Encoder Loss: ', round(E_loss.item(), 4), '\tLatent Discriminator Loss: ', round(d_latent_loss.item(), 4))

    fake_z_sample = fake_z[0].cpu().detach().numpy()
    u = fake_z_sample.mean()
    std = fake_z_sample.std()
    kstest = stats.kstest(fake_z_sample, 'norm', (u, std))
    print('pvalue (latent space vs gaussian distribution): ' + str(kstest.pvalue))
    print('=====================================================================\n')

writer.close()

torch.save(generator.state_dict(), 'generator_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_spec.state_dict(), 'dis_spec_' + str(FEAT_DIM) + 'd.pt')
torch.save(encoder.state_dict(), 'encoder_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_latent.state_dict(), 'dis_latent_' + str(FEAT_DIM) + 'd.pt')