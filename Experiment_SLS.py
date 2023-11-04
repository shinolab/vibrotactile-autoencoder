'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-07-04 01:27:58
LastEditors: Mingxin Zhang
LastEditTime: 2023-11-04 17:49:22
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
from GlobalOptimizer import JacobianOptimizer
from CAAE_14_norm import model
import torch
import pickle
import sys
import scipy
import librosa
import sounddevice as sd
import pyloudnorm as pyln
import pySequentialLineSearch
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, 
                             QWidget, QSlider, QPushButton, QLabel, QFrame)
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


device = torch.device("cpu")
print(f'Selected device: {device}')

FEAT_DIM = 128
CLASS_NUM = 14

def img_denormalize(img):
    # Min of original data: -80
    # Max of original data: 0
    origin_max = 0.
    origin_min = -80.
    img = (img + 1) / 2 # from [-1, 1] back to [0, 1]
    denormalized_img = img * (origin_max - origin_min) + origin_min
    return denormalized_img


class HeatmapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vibration Optimizer")
        self.setGeometry(100, 100, 200, 400)

        model_name = 'CAAE_14_norm'
        self.decoder = model.Generator(feat_dim=FEAT_DIM)

        # Model initialization and parameter loading
        decoder_dict = torch.load(model_name + '/generator_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cpu'))
        decoder_dict = {k: v for k, v in decoder_dict.items()}
        self.decoder.load_state_dict(decoder_dict)

        self.decoder.eval() 
        self.decoder.to(device)

        with open('trainset_7-class.pickle', 'rb') as file:
            trainset = pickle.load(file)
    
        index = np.random.randint(len(trainset['spectrogram']))
        self.target_spec = trainset['spectrogram'][index]

        self.target_wav = self.spec2wav(self.target_spec)
        meter = pyln.Meter(44100) # create BS.1770 meter
        self.target_loudness = meter.integrated_loudness(self.target_wav)

        self.optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=6)

        self.optimizer.set_hyperparams(kernel_signal_var=0.50,
                                kernel_length_scale=0.10,
                                kernel_hyperparams_prior_var=0.10)
        
        self.optimizer.set_gaussian_process_upper_confidence_bound_hyperparam(5.)

        self.pca = joblib.load('PCA_z6.m')

        self.initUI()

    def initUI(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout(main_widget)

        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)

        target_title = QLabel('Target Vibration Recording')
        target_title.setFont(title_font)
        layout.addWidget(target_title, 1, Qt.AlignCenter | Qt.AlignTop)

        real_vib_layout = QHBoxLayout()
        real_vib_layout.addWidget(QLabel('Click to play the vibration'), 1, Qt.AlignCenter | Qt.AlignCenter)

        play_stop_button = QPushButton("Play")
        play_stop_button.clicked.connect(self.playRealVib)
        real_vib_layout.addWidget(play_stop_button, 1, Qt.AlignCenter | Qt.AlignCenter)

        self.wav_gif = QMovie('UI/giphy.gif')

        self.wav_gif = QMovie()
        self.wav_gif.setFileName('UI/giphy.gif')
        self.wav_gif.jumpToFrame(0)

        self.gif_label = QLabel()
        self.gif_label.setMovie(self.wav_gif)
        self.gif_label.setMinimumSize(QtCore.QSize(180, 75))
        self.gif_label.setMaximumSize(QtCore.QSize(180, 150))
        self.gif_label.setScaledContents(True)

        layout.addWidget(self.gif_label, 1, Qt.AlignCenter | Qt.AlignCenter)
        layout.addLayout(real_vib_layout)
        layout.addSpacing(18)

        optimization_title = QLabel('Generated Vibration')
        optimization_title.setFont(title_font)
        layout.addWidget(optimization_title, 1, Qt.AlignCenter | Qt.AlignTop)

        # 创建滑块并设置范围和初始值
        layout.addWidget(QLabel('Select the slider position of\
                                \nthe best matching vibration'), 1, Qt.AlignCenter | Qt.AlignTop)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 1000)
        self.slider.setValue(500)
        layout.addWidget(self.slider)

        next_save_button = QHBoxLayout()
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda value: self.updateValues(_update_optimizer_flag=True))
        next_save_button.addWidget(next_button)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.saveWavFile)
        next_save_button.addWidget(save_button)

        layout.addWidget(QLabel('Click \'Next\' to enter the next iteration\
                                \nClick \'Save\' to save the current vibration'), 1, Qt.AlignCenter | Qt.AlignTop)

        layout.addLayout(next_save_button)

        # 连接滑块的valueChanged信号到更新热度图的槽函数
        self.slider.valueChanged.connect(lambda value: self.updateValues(_update_optimizer_flag=False))

        self.updateValues(_update_optimizer_flag=False)
        sd.stop()

    def spec2wav(self, spec):
        ex = np.full((1025 - spec.shape[0], spec.shape[1]), -80) #もとの音声の周波数上限を切っているので配列の大きさを合わせるために-80dbで埋めている
        spec = np.append(spec, ex, axis=0)

        spec = librosa.db_to_amplitude(spec)
        re_wav = librosa.griffinlim(spec,n_iter=100, n_fft=2048, hop_length=int(2048 * 0.1), window='hann')

        return re_wav

    def playRealVib(self):
        play_wav = pyln.normalize.loudness(self.target_wav, self.target_loudness, -12.)

        play_wav = np.tile(play_wav, 10)
        sd.play(play_wav, samplerate=44100)
        self.wav_gif.start()

    def saveWavFile(self):
        scipy.io.wavfile.write("Generated.wav", 44100, self.re_wav)

    def updateValues(self, _update_optimizer_flag):
        # 获取滑块的值
        slider_value = self.slider.value()
        t = slider_value / 999

        if _update_optimizer_flag:
            self.optimizer.submit_feedback_data(t)
            print('Next')

        # range of real latent space: [-7.24, 6.42]
        # range of real pca_z: [-8.34 18.69]

        z_low = self.optimizer.calc_point_from_slider_position(t)

        z_low_max = 18.69
        z_low_min = -8.34
        z_low = z_low * (z_low_max - z_low_min) + z_low_min

        z = self.pca.inverse_transform(z_low)

        z = torch.unsqueeze(torch.tensor(z).to(torch.float32), 0)
        z = z.to(device)

        spec = self.decoder(z).cpu().detach().numpy().reshape(48, 320)

        re_wav = self.spec2wav(spec)

        meter = pyln.Meter(44100) # create BS.1770 meter
        loudness = meter.integrated_loudness(re_wav)

        # loudness normalize audio to target
        loudness_normalized_audio = pyln.normalize.loudness(re_wav, loudness, -12.)

        self.re_wav = np.tile(loudness_normalized_audio, 10)
        sd.play(self.re_wav)

        self.wav_gif.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeatmapWindow()
    window.show()
    sys.exit(app.exec_())
