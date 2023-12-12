'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-07-04 01:27:58
LastEditors: Mingxin Zhang
LastEditTime: 2023-12-07 16:57:02
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
from CAAE_14class import model
import torch
import pickle
import sys
import torchaudio
import Methods
import UserInterface
from PyQt5.QtWidgets import QApplication


device = torch.device("cuda")
print(f'Selected device: {device}')

FEAT_DIM = 128
CLASS_NUM = 14
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

    model_name = 'CAAE_14class'
    decoder = model.Generator(feat_dim=FEAT_DIM)
    decoder.eval() 
    decoder.to(device)

    # Model initialization and parameter loading
    decoder_dict = torch.load(model_name + '/generator_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cuda'))
    decoder_dict = {k: v for k, v in decoder_dict.items()}
    decoder.load_state_dict(decoder_dict)

    target_latent = np.random.uniform(-2.5, 2.5, FEAT_DIM)
    target_latent = torch.tensor(target_latent).to(torch.float32).to(device)

    while True:
        random_A = Methods.getRandomAMatrix(FEAT_DIM, 6, np.array(target_latent.reshape(1, -1).cpu()), 1)
        if random_A is not None:
            break
    # random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)

    # initialize the latent
    init_z = np.random.uniform(low=-2.5, high=2.5, size=(FEAT_DIM))
    init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T
    init_z = np.matmul(random_A, init_low_z)

    app = QApplication(sys.argv)
    window = UserInterface.DSS_Experiment(griffinlim, 
                                          target_spec, 
                                          decoder, 
                                          init_z)
    window.show()
    sys.exit(app.exec_())
