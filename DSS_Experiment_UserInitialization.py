'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-07-04 01:27:58
LastEditors: Mingxin Zhang
LastEditTime: 2023-12-07 15:15:25
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
import torch
import pickle
import sys
import torchaudio
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

    with open('testset_7-class.pickle', 'rb') as file:
            testset = pickle.load(file)
    
    index = np.random.randint(len(testset['spectrogram']))
    target_spec = testset['spectrogram'][index]
    print(testset['filename'][index])
    group = testset['filename'][index][:2]

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
                                           target_spec, 
                                           decoder, 
                                           init_z,
                                           'Experiment')
    init_window.show()
    sys.exit(app.exec_())
