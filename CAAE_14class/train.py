'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-06-28 03:44:36
LastEditors: Mingxin Zhang
LastEditTime: 2023-11-03 03:17:31
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import numpy as np
import torch.optim as optim
import model
import time
import torch
import pickle
import os
from sklearn import preprocessing
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from scipy import stats

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'trainset_14-class.pickle')
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
    batch_size = 128,
    shuffle = True,
    num_workers = 0,
    )

adversarial_loss = nn.BCELoss()
classifier_loss = nn.CrossEntropyLoss()
image_loss = nn.MSELoss()

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

tv_loss = TVLoss()

FEAT_DIM = 128
CLASS_NUM = 14
encoder = model.ResNetEncoder(feat_dim = FEAT_DIM)
generator= model.Generator(feat_dim = FEAT_DIM)
cla_latent = model.LatentClassifier(feat_dim = FEAT_DIM, class_dim = CLASS_NUM)
dis_spec = model.SpectrogramDiscriminator(class_dim = CLASS_NUM)

gen_lr = 2e-4
d_spec_lr = 2e-4
c_lr = 1e-3

params_classifier = [
    {'params': encoder.parameters()},
    {'params': cla_latent.parameters()}
]

optimizer_G = optim.Adam(generator.parameters(), lr=gen_lr)
optimizer_D_spec = optim.Adam(dis_spec.parameters(), lr=d_spec_lr)
optimizer_C = optim.Adam(params_classifier, lr=c_lr)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer_C, gamma=0.95)

encoder.to(device)
cla_latent.to(device)
generator.to(device)
dis_spec.to(device)

epoch_num = 50

writer = SummaryWriter()
tic = time.time()
batch_num = 0

for epoch in range(1, epoch_num + 1):
    encoder.train()
    cla_latent.train()
    generator.train()
    dis_spec.train()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    for i, (img, label) in enumerate(train_dataloader):
        batch_num += 1

        img = torch.unsqueeze(img, 1) # Add channel axis (1 channel)
        img = img.to(device)
        label = label.to(torch.float32).to(device)

        soft_scale = 0.3
        valid = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        fake = torch.autograd.Variable(torch.Tensor(img.size(0), 1).fill_(0.0), requires_grad=False).to(device)

        # 1) reconstruction
        # 1.1) generator
        optimizer_G.zero_grad()
        # input latent vector
        z = encoder(img)
        # train generator]
        gen_img = generator(z)
        output_d = dis_spec(gen_img, label)

        g_loss = adversarial_loss(output_d, valid) + 100 * image_loss(gen_img, img) + 10 * tv_loss(gen_img)
        g_loss.backward()
        optimizer_G.step()

        writer.add_scalar('Spectrogram/G_loss', g_loss.item(), batch_num)

        # 1.2) spectrogram discriminator
        soft_valid = valid - torch.rand(img.size(0), 1).to(device) * soft_scale
        soft_fake = fake + torch.rand(img.size(0), 1).to(device) * soft_scale

        optimizer_D_spec.zero_grad()
        z = encoder(img)
        gen_img = generator(z)
        
        # loss for real img
        output_d = dis_spec(img, label)
        real_loss = adversarial_loss(output_d, soft_valid)

        # loss for fake img
        output_d = dis_spec(gen_img.detach(), label)
        fake_loss = adversarial_loss(output_d, soft_fake)

        d_spec_loss = (real_loss + fake_loss) / 2

        d_spec_loss.backward()
        optimizer_D_spec.step()

        writer.add_scalar('Spectrogram/D_loss', d_spec_loss.item(), batch_num)

        # 2) latent classifier
        optimizer_C.zero_grad()
        z = encoder(img)

        output_c = cla_latent(z)

        C_loss = classifier_loss(output_c, label)
        C_loss.backward()
        optimizer_C.step()

        prediction = torch.argmax(output_c, 1)
        correct += (prediction == torch.argmax(label, 1)).sum().float()
        total += len(label)

        writer.add_scalar('Latent/C_loss', C_loss.item(), batch_num)

    toc = time.time()

    scheduler.step()

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
    print('Classification Loss: ', round(C_loss.item(), 4))
    print('Accuracy: ', ((correct / total).cpu().detach().data.numpy()))
    z_sample = z[0].cpu().detach().numpy()
    u = z_sample.mean()
    std = z_sample.std()
    kstest = stats.kstest(z_sample, 'norm', (u, std))
    print('z_u: ', u, 'z_std: ', std)
    print('pvalue (latent space vs gaussian distribution): ' + str(kstest.pvalue))
    print('=====================================================================\n')

writer.close()

torch.save(generator.state_dict(), 'generator_' + str(FEAT_DIM) + 'd.pt')
torch.save(dis_spec.state_dict(), 'dis_spec_' + str(FEAT_DIM) + 'd.pt')
torch.save(encoder.state_dict(), 'encoder_' + str(FEAT_DIM) + 'd.pt')

# Training classification acc: 93.05%