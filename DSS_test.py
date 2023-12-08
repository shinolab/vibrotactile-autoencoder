'''
Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
Date: 2023-04-12 01:47:50
LastEditors: Mingxin Zhang
LastEditTime: 2023-12-08 17:44:40
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

from GlobalOptimizer import JacobianOptimizer
from CAAE_14class import model
import torch
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("mps")
device = torch.device("cuda")
print(f'Selected device: {device}')

FEAT_DIM = 128
CLASS_NUM = 14
SLIDER_LEN = 30

def img_denormalize(img):
    # Min of original data: -80
    # Max of original data: 0
    origin_max = 0.
    origin_min = -80.
    img = (img + 1) / 2 # from [-1, 1] back to [0, 1]
    denormalized_img = img * (origin_max - origin_min) + origin_min
    return denormalized_img

def z_denormalize(z):
    # range of real latent space: [-7.24, 6.42]
    origin_max = 6.42
    origin_min = -7.24
    z = (z + 1) / 2
    denormalized_z = z * (origin_max - origin_min) + origin_min
    return denormalized_z

def myFunc(decoder, zs):
    zs = z_denormalize(zs)
    zs = torch.tensor(zs).to(torch.float32).to(device)
    output = img_denormalize(decoder(zs)).reshape(zs.shape[0], -1)
    # output = decoder(zs).reshape(zs.shape[0], -1)
    return output

def myGoodness(target, xs):
    xs = torch.tensor(xs).to(torch.float32).to(device)
    return np.sum((xs.reshape(xs.shape[0], -1) - target.reshape(1, -1)).cpu().detach().numpy() ** 2, axis=1) ** 0.5

def myJacobian(model, z):
    z = z_denormalize(z)
    z = torch.tensor(z).to(torch.float32).to(device)
    output = model.calc_model_gradient(z, device)
    return output

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
    model_name = 'CAAE_14class'
    decoder = model.Generator(feat_dim=FEAT_DIM)

    result_dict = {'original': [], 
                  'generated': [], 
                  'soundfile': []
                }

    # Model initialization and parameter loading
    decoder_dict = torch.load(model_name + '/generator_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cpu'))
    decoder_dict = {k: v for k, v in decoder_dict.items()}
    decoder.load_state_dict(decoder_dict)

    decoder.eval() 
    decoder.to(device)

    # with open('sample_target_spec_1.pickle', 'rb') as file:
    #     target_spec = pickle.load(file)

    with open('testset_7-class.pickle', 'rb') as file:
        testset = pickle.load(file)

    test_index = [68, 2248, 2947, 3893, 4181, 5437, 10451, 11387, 13158, 14095]
    
    iter_num = 20
    
    avg_loss = np.zeros(iter_num+1)

    # repeat generation experiments
    for index in test_index:
        target_spec = testset['spectrogram'][index]
        soundfile = testset['filename'][index]

        target_data = torch.unsqueeze(torch.tensor(target_spec), 0).to(torch.float32).to(device)

        slider_length = getSliderLength(FEAT_DIM, 1, 0.8)
        target_latent = np.random.uniform(-2.5, 2.5, FEAT_DIM)
        target_latent = torch.tensor(target_latent).to(torch.float32).to(device)
        # target_data = decoder(target_latent.reshape(1, -1))[0]

        while True:
            random_A = getRandomAMatrix(FEAT_DIM, 6, np.array(target_latent.reshape(1, -1).cpu()), 1)
            if random_A is not None:
                break
        # random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)
        
        init_z = np.random.uniform(low=-2.5, high=2.5, size=(FEAT_DIM))
        init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T
        init_z = np.matmul(random_A, init_low_z)

        optimizer = JacobianOptimizer.JacobianOptimizer(FEAT_DIM, 48*320, 
                        lambda zs: myFunc(decoder, zs), 
                        lambda xs: myGoodness(target_data, xs), 
                        slider_length, 
                        lambda z: myJacobian(decoder, z), 
                        maximizer=False)

        optimizer.init(init_z)
        best_score = optimizer.current_score
        avg_loss[0] += best_score

        imshape = (48, 320)
        imdata = np.zeros(imshape)
        
        for i in range(iter_num):
            n_sample = SLIDER_LEN
            opt_z, opt_x, opt_score, opt_t = optimizer.find_optimal(n_sample, batch_size=n_sample)
            # print(opt_z)
            if opt_score < best_score:
                best_score = opt_score

            print('Iteration #' + str(i) + ': ' + str(best_score))
            avg_loss[i+1] += best_score
            optimizer.update(opt_t)
    
    avg_loss /= len(test_index)
    for loss in avg_loss:
        print(loss)

if __name__ == '__main__':
    main()

