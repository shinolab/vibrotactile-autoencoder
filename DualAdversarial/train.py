'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:44:36
LastEditors: Mingxin Zhang
LastEditTime: 2023-06-29 09:26:16
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''


import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from sklearn import preprocessing
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from scipy import stats
import torch.nn.functional as F
import torch.optim as optim
import model
import time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

with open('trainset_LMT_large.pickle', 'rb') as file:
# with open('trainset.pickle', 'rb') as file:
    trainset = pickle.load(file)

spectrogram = torch.from_numpy(trainset['spectrogram'].astype(np.float32))
texture = trainset['texture']
le = preprocessing.LabelEncoder()
labels = torch.as_tensor(le.fit_transform(texture))

transform_spec = transforms.Normalize(
    mean = spectrogram.mean(),
    std = spectrogram.std()
)

spectrogram = transform_spec(spectrogram)

train_dataset = torch.utils.data.TensorDataset(spectrogram, labels)
train_dataloader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True,
    num_workers = 0,
    )

adversarial_loss = nn.BCELoss()

FEAT_DIM = 128
encoder = model.ResNetEncoder(encoded_space_dim = FEAT_DIM)
generator= model.Generator(encoded_space_dim = FEAT_DIM)
dis_latent = model.LatentDiscriminator(encoded_space_dim = FEAT_DIM)
dis_spec = model.SpectrogramDiscriminator()

gen_lr = 2e-4
encoder_lr = 2e-4
d_spec_lr = 2e-4
d_latent_lr = 2e-4

optimizer_G = optim.Adam(generator.parameters(), lr=gen_lr)
optimizer_E = optim.Adam(encoder.parameters(), lr=encoder_lr)
optimizer_D_spec = optim.Adam(dis_spec.parameters(), lr=d_spec_lr)
optimizer_D_latent = optim.Adam(dis_latent.parameters(), lr=d_latent_lr)

encoder.to(device)
dis_latent.to(device)
generator.to(device)
dis_spec.to(device)

epoch_num = 150

EPS = 1e-15

tic = time.time()
for epoch in range(1, epoch_num + 1):
    encoder.train()
    dis_latent.train()
    generator.train()
    dis_spec.train()

    for i, (img, label) in enumerate(train_dataloader):
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
        optimizer_G.zero_grad()
        # input latent vector
        z = encoder(img)
        # train generator
        gen_img = generator(z)
        g_loss = adversarial_loss(dis_spec(gen_img), soft_valid)
        g_loss.backward()
        optimizer_G.step()

        # 1.2) spectrogram discriminator
        optimizer_D_spec.zero_grad()

        s1, s2, s3, s4 = img.shape
        means = torch.zeros(s1,s2,s3,s4)
        std = torch.ones(s1,s2,s3,s4)

        sigma = 1
        std = std * sigma

        noise_r = torch.normal(means, std).to(device)
        noise_g = torch.normal(means, std).to(device)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(dis_spec(img + noise_r), soft_valid)
        fake_loss = adversarial_loss(dis_spec(gen_img.detach() + noise_g), soft_fake)
        d_spec_loss = (real_loss + fake_loss) / 2

        d_spec_loss.backward()
        optimizer_D_spec.step()
        optimizer_E.step()

        # 2) latent discriminator
        real_z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (img.shape[0], FEAT_DIM)))).to(device)
        fake_z = encoder(img)

        # D_real_gauss = dis_latent(real_z)
        # D_fake_gauss = dis_latent(fake_z.detach())
        # d_latent_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

        real_loss = adversarial_loss(dis_latent(real_z), valid)
        fake_loss = adversarial_loss(dis_latent(fake_z.detach()), fake)
        d_latent_loss = (real_loss + fake_loss) / 2
        d_latent_loss.backward()
        optimizer_D_latent.step()

        # 3) encoder
        fake_z = encoder(img)

        # D_fake_gauss = dis_latent(fake_z)
        # E_loss = -torch.mean(torch.log(D_fake_gauss + EPS))

        E_loss = adversarial_loss(dis_latent(fake_z), valid)
        E_loss.backward()
        optimizer_E.step()

    toc = time.time()
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

torch.save(generator.state_dict(), 'weights/generator_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_spec.state_dict(), 'weights/dis_spec_' + str(FEAT_DIM) + 'd.pt')
torch.save(encoder.state_dict(), 'weights/encoder_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_latent.state_dict(), 'weights/dis_latent_' + str(FEAT_DIM) + 'd.pt')