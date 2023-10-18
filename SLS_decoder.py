'''
Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
Date: 2023-04-12 01:47:50
LastEditors: Mingxin Zhang
LastEditTime: 2023-10-18 17:39:48
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import pySequentialLineSearch
from model import Autoencoder
from model import VAE
from model import AAE
from model import GAN
import torch
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import MinMaxScaler

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("mps")
device = torch.device("cpu")
print(f'Selected device: {device}')

FEAT_DIM = 8

# A dummy implementation of slider manipulation
def ask_human_for_slider_manipulation(optimizer, decoder, target_spec, norm_scaler=False):
    t_max = 0.0
    f_max = -sys.float_info.max

    mse = nn.MSELoss()

    for i in range(1000):
        
        t = float(i) / 999.0

        generated_vector = optimizer.calc_point_from_slider_position(t)

        # denormalization
        generated_vector = torch.unsqueeze(torch.tensor(generated_vector), 0)
        if norm_scaler != False:
            generated_vector = norm_scaler.inverse_transform(generated_vector)
        generated_vector = torch.tensor(generated_vector).to(torch.float32).to(device)
        decoded_spec = decoder(generated_vector)

        f = -mse(decoded_spec, target_spec).item()

        if f_max is None or f_max < f:
            f_max = f
            t_max = t

    return t_max


def main():
    model = input('Choose generation model:' + '\n' + \
          '[1]Autoencoder' + '\n'\
          '[2]Variational Autoencoder' + '\n'\
          '[3]Adversarial Autoencoder' + '\n'\
          '[4]Generative Adversarial Network' + '\n')
    
    if model == '1':
        model_name = 'Autoencoder'
        decoder = Autoencoder.Decoder(encoded_space_dim = FEAT_DIM)
    if model == '2':
        model_name = 'VAE'
        decoder = VAE.Decoder(encoded_space_dim = FEAT_DIM)
    if model == '3':
        model_name = 'AAE'
        decoder = AAE.Decoder(encoded_space_dim = FEAT_DIM)
    if model == '4':
        model_name = 'GAN'
        decoder = GAN.Generator(encoded_space_dim = FEAT_DIM)

    # Model initialization and parameter loading
    if model_name != 'GAN':
        decoder_dict = torch.load('model/' + model_name + '/decoder_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cpu'))
    else:
        decoder_dict = torch.load('model/' + model_name + '/generator_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cpu'))
    decoder_dict = {k: v for k, v in decoder_dict.items()}
    decoder.load_state_dict(decoder_dict)

    decoder.eval()
    decoder.to(device)

    with open('sample_target_spec_2.pickle', 'rb') as file:
        target_spec = pickle.load(file)
    # plt.imshow(target_spec)
    # plt.show()
    target_spec = np.expand_dims(target_spec, axis=0)

    # Load the extracted n-dimensional features
    if model_name != 'GAN':
        with open('feat_dict/' + model_name + '/feat_dict_' + str(FEAT_DIM) + 'd.pickle', 'rb') as file:
            feat_dict = pickle.load(file)

        vib_feat = feat_dict['vib_feat']
        norm_scaler = MinMaxScaler()
        norm_vib_feat = norm_scaler.fit_transform(vib_feat) # normalization

        # Randomly choose a target spectrogram
        # original_vib = feat_dict['original_vib']
        # target_index = np.random.randint(0, len(original_vib))
        # target_spec = original_vib[target_index]
    
    else:
        with open('feat_dict/VAE/feat_dict_' + str(FEAT_DIM) + 'd.pickle', 'rb') as file:
            feat_dict = pickle.load(file)
        
        # Randomly choose a target spectrogram
        # original_vib = feat_dict['original_vib']
        # target_index = np.random.randint(0, len(original_vib))
        # target_spec = original_vib[target_index]
    
    target_spec = torch.unsqueeze(torch.tensor(target_spec), 0).to(device)

    optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
        num_dims=FEAT_DIM)

    optimizer.set_hyperparams(kernel_signal_var=0.50,
                              kernel_length_scale=0.10,
                              kernel_hyperparams_prior_var=0.10)
    
    optimizer.set_gaussian_process_upper_confidence_bound_hyperparam(5.)

    for i in range(20):
        # slider_ends = optimizer.get_slider_ends()
        if model_name != 'GAN':
            slider_position = ask_human_for_slider_manipulation(optimizer, decoder, target_spec, norm_scaler)
        else:
            slider_position = ask_human_for_slider_manipulation(optimizer, decoder, target_spec)
        optimizer.submit_feedback_data(slider_position)

        optimized_vector = optimizer.get_maximizer()
        # denormalization
        optimized_vector = torch.unsqueeze(torch.tensor(optimized_vector), 0)

        if model_name != 'GAN':
            optimized_vector = torch.tensor(norm_scaler.inverse_transform(optimized_vector)).to(torch.float32)
        else:
            optimized_vector = optimized_vector.to(torch.float32)

        optimized_vector = optimized_vector.to(device)
        decoded_spec = decoder(optimized_vector)

        mse = nn.MSELoss()
        loss = mse(decoded_spec, target_spec).item()
        print("[#iters = " + str(i + 1) + "], slider_position: " + str(slider_position) + ", loss: " + str(loss))
    
    plt.imshow(decoded_spec.squeeze().detach().numpy())
    plt.show()


if __name__ == '__main__':
    main()

