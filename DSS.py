'''
Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
Date: 2023-04-12 01:47:50
LastEditors: Mingxin Zhang
LastEditTime: 2023-07-04 01:14:56
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import pySequentialLineSearch
from GlobalOptimizer import JacobianOptimizer
from SRResNet_ACGAN import model
import torch
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("mps")
device = torch.device("cpu")
print(f'Selected device: {device}')

FEAT_DIM = 256

def denormalize(img):
    # Min of original data: -80
    # Max of original data: 0
    origin_max = 0.
    origin_min = -80.
    img = (img + 1) / 2 # from [-1, 1] back to [0, 1]
    denormalized_img = img * (origin_max - origin_min) + origin_min
    return denormalized_img

def myFunc(decoder, zs):
    zs = torch.tensor(zs).to(torch.float32).to(device)
    output = denormalize(decoder(zs)).reshape(zs.shape[0], -1)
    # output = decoder(zs).reshape(zs.shape[0], -1)
    return output

def myGoodness(target, xs):
    xs = torch.tensor(xs).to(torch.float32).to(device)
    return np.sum((xs.reshape(xs.shape[0], -1) - target.reshape(1, -1)).cpu().detach().numpy() ** 2, axis=1) ** 0.5

def myJacobian(model, z):
    z = torch.tensor(z).to(torch.float32).to(device)
    return model.calc_model_gradient(z, device)

def getSliderLength(n, boundary_range, ratio, sample_num=1000):
    samples = np.random.uniform(low=-boundary_range, high=boundary_range, size=(sample_num, 2, n))
    distances = np.linalg.norm(samples[:, 0, :] - samples[:, 1, :], axis=1)
    average_distance = np.average(distances)
    return ratio * average_distance

def getRandomAMatrix(high_dim, dim, optimals, range):
    A = np.random.normal(size=(high_dim, dim))
    try:
        invA = np.linalg.pinv(A)
    except:
        print("Inverse failed!")
        return None

    low_optimals = np.matmul(invA, optimals.T).T
    conditions = (low_optimals < range) & (low_optimals > -range)
    conditions = np.all(conditions, axis=1)
    if np.any(conditions):
        return A
    else:
        print("A matrix is not qualified. Resampling......")
        return None

def main():
    model_name = 'SRResNet_ACGAN'
    decoder = model.Generator(encoded_space_dim = FEAT_DIM)

    # Model initialization and parameter loading
    decoder_dict = torch.load(model_name + '/generator_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cpu'))
    decoder_dict = {k: v for k, v in decoder_dict.items()}
    decoder.load_state_dict(decoder_dict)

    decoder.eval()
    decoder.to(device)

    with open('sample_target_spec_1.pickle', 'rb') as file:
        target_spec = pickle.load(file)

    target_data = torch.unsqueeze(torch.tensor(target_spec), 0).to(torch.float32).to(device)

    slider_length = getSliderLength(FEAT_DIM, 1, 0.2)
    target_latent = np.random.uniform(-1, 1, FEAT_DIM)
    target_latent = torch.tensor(target_latent).to(torch.float32).to(device)
    # target_data = decoder(target_latent.reshape(1, -1))[0]

    while True:
        random_A = getRandomAMatrix(FEAT_DIM, 6, np.array(target_latent.reshape(1, -1).cpu()), 1)
        if random_A is not None:
            break
    # random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)
    
    init_z = np.random.uniform(low=-1, high=1, size=(FEAT_DIM))
    init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T
    init_z = np.matmul(random_A, init_low_z)

    print(slider_length)

    optimizer = JacobianOptimizer.JacobianOptimizer(FEAT_DIM, 12*160, 
                      lambda zs: myFunc(decoder, zs), 
                      lambda xs: myGoodness(target_data, xs), 
                      slider_length, 
                      lambda z: myJacobian(decoder, z), 
                      maximizer=False)

    optimizer.init(init_z)
    best_score = optimizer.current_score

    iter_num = 20
    
    for i in range(iter_num):
        n_sample = 1000
        opt_z, opt_x, opt_score, opt_t = optimizer.find_optimal(n_sample, batch_size=n_sample)
        if opt_score < best_score:
            best_score = opt_score

        print('Iteration #' + str(i) + ': ' + str(best_score))
        optimizer.update(opt_t)

    fig, ax = plt.subplots(2, 1, figsize=(5, 3)) 
    fig.suptitle('Iter num = ' + str(iter_num) + ', loss = ' + str(best_score), fontsize=16)
    ax[0].imshow(target_spec.reshape(12, 160)) 
    ax[0].set_title("Original") 
    ax[1].imshow(opt_x.cpu().detach().numpy().reshape(12, 160)) 
    ax[1].set_title("Generated")
    plt.show()


if __name__ == '__main__':
    main()

