'''
Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
Date: 2023-04-12 01:47:50
LastEditors: Mingxin Zhang
LastEditTime: 2023-04-15 02:48:44
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import pySequentialLineSearch
import Autoencoder
import torch
import pickle
import numpy as np
import sys
from torch import nn
from sklearn.preprocessing import MinMaxScaler

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("mps")
device = torch.device("cpu")
print(f'Selected device: {device}')

# A dummy implementation of slider manipulation
def ask_human_for_slider_manipulation(optimizer, decoder, norm_scaler, target_spec):
    t_max = 0.0
    f_max = -sys.float_info.max

    mse = nn.MSELoss()

    for i in range(1000):
        
        t = float(i) / 999.0

        generated_vector = optimizer.calc_point_from_slider_position(t)

        # denormalization
        generated_vector = torch.unsqueeze(torch.tensor(generated_vector), 0)
        generated_vector = torch.tensor(norm_scaler.inverse_transform(generated_vector)).to(torch.float32)
        generated_vector = generated_vector.to(device)
        decoded_spec = decoder(generated_vector)

        f = -mse(decoded_spec, target_spec).item()

        if f_max is None or f_max < f:
            f_max = f
            t_max = t

    return t_max


def main():
    # Model initialization and parameter loading
    # TestDecoder is used for test without indices of pooling
    decoder = Autoencoder.TestDecoder(encoded_space_dim = 20)
    # decoder_dict = torch.load('/content/drive/MyDrive/Colab Notebooks/vibrotactile-encoder/decoder.pt', map_location=torch.device('cpu'))
    decoder_dict = torch.load('decoder.pt', map_location=torch.device('cpu'))
    decoder_dict = {k: v for k, v in decoder_dict.items()}
    decoder.load_state_dict(decoder_dict)

    decoder.eval()
    decoder.to(device)

    # Load the extracted 20-dimensional features
    with open('feat_dict.pickle', 'rb') as file:
        feat_dict = pickle.load(file)

    vib_feat = feat_dict['vib_feat']
    original_vib = feat_dict['original_vib']

    # Randomly choose a target spectrogram
    target_index = np.random.randint(0, len(original_vib))
    target_spec = original_vib[target_index]
    target_spec = torch.unsqueeze(torch.tensor(target_spec), 0).to(device)

    norm_scaler = MinMaxScaler()
    norm_vib_feat = norm_scaler.fit_transform(vib_feat) # normalization

    optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(
        num_dims=20)

    optimizer.set_hyperparams(kernel_signal_var=0.50,
                              kernel_length_scale=0.10,
                              kernel_hyperparams_prior_var=0.10)
    
    optimizer.set_gaussian_process_upper_confidence_bound_hyperparam(5.)

    for i in range(10):
        # slider_ends = optimizer.get_slider_ends()
        slider_position = ask_human_for_slider_manipulation(optimizer, decoder, norm_scaler, target_spec)
        optimizer.submit_feedback_data(slider_position)

        optimized_vector = optimizer.get_maximizer()
        # denormalization
        optimized_vector = torch.unsqueeze(torch.tensor(optimized_vector), 0)
        optimized_vector = torch.tensor(norm_scaler.inverse_transform(optimized_vector)).to(torch.float32)
        optimized_vector = optimized_vector.to(device)
        decoded_spec = decoder(optimized_vector)

        mse = nn.MSELoss()
        loss = mse(decoded_spec, target_spec).item()
        print("[#iters = " + str(i + 1) + "], slider_position: " + str(slider_position) + ", loss: " + str(loss))


if __name__ == '__main__':
    main()

