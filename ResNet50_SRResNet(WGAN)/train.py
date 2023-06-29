'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:44:36
LastEditors: Mingxin Zhang
LastEditTime: 2023-06-28 03:51:16
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

gen_lr = 1e-4
encoder_lr = 5e-5
d_spec_lr = 1e-4
d_latent_lr = 5e-5

optimizer_G = optim.RMSprop(generator.parameters(), lr=gen_lr)
optimizer_E = optim.RMSprop(encoder.parameters(), lr=encoder_lr)
optimizer_D_spec = optim.RMSprop(dis_spec.parameters(), lr=d_spec_lr)
optimizer_D_latent = optim.RMSprop(dis_latent.parameters(), lr=d_latent_lr)

encoder.to(device)
dis_latent.to(device)
generator.to(device)
dis_spec.to(device)

epoch_num = 150

writer = SummaryWriter(os.path.join(current_dir, 'runs', 'train'))

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

        valid = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        fake = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(0.0), requires_grad=False).to(device)

        # 1) reconstruction
        # 1.1) spectrogram discriminator
        # train netd more: because the better netd is,
        # the better netg will be
        for stepp in range(5):
            for parm in dis_spec.parameters():
                clamp_num = 0.01 # WGAN clip gradient
                parm.data.clamp_(-clamp_num, clamp_num)

            optimizer_D_spec.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_output = dis_spec(img)

            z = encoder(img)
            gen_img = generator(z).detach()
            fake_output = dis_spec(gen_img)

            d_spec_loss = -torch.mean(real_output - fake_output)
            d_spec_loss.backward()
            optimizer_D_spec.step()

        writer.add_scalar('Spectrogram/D_loss', d_spec_loss.item(), batch_num)

        # 1.2) generator
        optimizer_G.zero_grad()
        # input latent vector
        z = encoder(img)
        # train generator
        gen_img = generator(z)
        fake_output = dis_spec(gen_img)
        g_loss = -torch.mean(fake_output)
        g_loss.backward()

        optimizer_G.step()

        writer.add_scalar('Spectrogram/G_loss', g_loss.item(), batch_num)

        # 2) latent discriminator
        for stepp in range(5):
            for parm in dis_latent.parameters():
                clamp_num = 0.01 # WGAN clip gradient
                parm.data.clamp_(-clamp_num, clamp_num)

            optimizer_D_latent.zero_grad()

            real_z = torch.autograd.Variable(torch.Tensor(np.random.normal(0, 1, (img.shape[0], FEAT_DIM)))).to(device)
            fake_z = encoder(img)

            real_output = dis_latent(real_z)
            fake_output = dis_latent(fake_z.detach())
            d_latent_loss = -torch.mean(real_output - fake_output)
            d_latent_loss.backward()
            optimizer_D_latent.step()

        writer.add_scalar('Latent/D_loss', d_latent_loss.item(), batch_num)

        # 3) encoder
        optimizer_E.zero_grad()
        fake_z = encoder(img)

        E_loss = -torch.mean(dis_latent(fake_z))
        E_loss.backward()
        optimizer_E.step()

        writer.add_scalar('Latent/G_loss', E_loss.item(), batch_num)

    toc = time.time()

    writer.add_image('Real Spectrogram', img[0], epoch)
    writer.add_image('Fake Spectrogram', gen_img[0], epoch)

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

torch.save(generator.state_dict(), 'weights/generator_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_spec.state_dict(), 'weights/dis_spec_' + str(FEAT_DIM) + 'd.pt')
torch.save(encoder.state_dict(), 'weights/encoder_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_latent.state_dict(), 'weights/dis_latent_' + str(FEAT_DIM) + 'd.pt')