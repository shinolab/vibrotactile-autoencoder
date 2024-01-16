'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-07-04 01:27:58
LastEditors: Mingxin Zhang
LastEditTime: 2024-01-16 19:18:42
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
import torch
import pickle
import sys
import os
import torchaudio
import librosa
import UserInterface
from PyQt5.QtWidgets import QApplication
from CAAE_14class import model


device = torch.device("cuda")
print(f'Selected device: {device}')

FEAT_DIM = 128
SLIDER_LEN = 30


if __name__ == "__main__":
    griffinlim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=50, hop_length=int(2048 * 0.1), power=1.0)
    griffinlim = griffinlim.to(device)

    # with open('testset_7-class.pickle', 'rb') as file:
    #         testset = pickle.load(file)
    
    target_group = input("Input the target group: ")
    # while True:
    #     index = np.random.randint(len(testset['spectrogram']))
    #     target_spec = testset['spectrogram'][index]
    #     if testset['filename'][index][:2] == target_group:
    #         print(testset['filename'][index])
    #         break
    
    real_data_path = 'Reference_Waves/'
    
    real_file_list = []
    for root, dirs, files in os.walk(real_data_path):
        for name in files:
            real_file_list.append(os.path.join(root, name))
    
    while True:
        index = np.random.randint(len(real_file_list))
        if real_file_list[index].split('/')[-1][:4] == target_group:
            target_file_name = real_file_list[index].split('/')[-1][:4]
            target_vib, fs = librosa.load(real_file_list[index], sr=44100)
            print(target_file_name)
            break

    with open('CAAE_14class/latent_dict.pickle', 'rb') as file:
        latent_dict = pickle.load(file)
    
    index = np.random.randint(len(latent_dict['z']))
    init_z = latent_dict['z'][index]

    model_name = 'CAAE_14class'
    decoder = model.Generator(feat_dim=FEAT_DIM)
    decoder.eval() 
    decoder.to(device)

    # Model initialization and parameter loading
    decoder_dict = torch.load(model_name + '/generator_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cuda'))
    decoder_dict = {k: v for k, v in decoder_dict.items()}
    decoder.load_state_dict(decoder_dict)

    app = QApplication(sys.argv)
    init_window = UserInterface.InitWindow(griffinlim, 
                                           target_vib, 
                                           target_group,
                                           target_file_name,
                                           decoder, 
                                           init_z,
                                           'Experiment')
    init_window.show()
    sys.exit(app.exec_())
