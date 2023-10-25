'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-07-04 01:27:58
LastEditors: Mingxin Zhang
LastEditTime: 2023-10-26 02:57:21
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
from GlobalOptimizer import JacobianOptimizer
from CAAE_LMT108 import model
import torch
import pickle
import sys
import scipy
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, 
                             QWidget, QSlider, QPushButton, QLabel, QFrame)
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


device = torch.device("cpu")
print(f'Selected device: {device}')

FEAT_DIM = 256
CLASS_NUM = 108

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
    

class HeatmapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vibration Optimizer")
        self.setGeometry(100, 100, 200, 400)

        model_name = 'CAAE_LMT108'
        self.decoder = model.Generator(feat_dim=FEAT_DIM, class_dim=CLASS_NUM)

        # Model initialization and parameter loading
        decoder_dict = torch.load(model_name + '/generator_' + str(FEAT_DIM) + 'd_gamma5.pt', map_location=torch.device('cpu'))
        decoder_dict = {k: v for k, v in decoder_dict.items()}
        self.decoder.load_state_dict(decoder_dict)

        self.decoder.eval() 
        self.decoder.to(device)

        with open('trainset_LMT_large.pickle', 'rb') as file:
            trainset = pickle.load(file)
    
        index = np.random.randint(len(trainset['spectrogram']))
        self.target_spec = trainset['spectrogram'][index]

        target_data = torch.unsqueeze(torch.tensor(self.target_spec), 0).to(torch.float32).to(device)

        slider_length = getSliderLength(FEAT_DIM+CLASS_NUM, 1, 0.2)
        target_latent = np.random.uniform(-1, 1, FEAT_DIM+CLASS_NUM)
        target_latent = torch.tensor(target_latent).to(torch.float32).to(device)

        while True:
            random_A = getRandomAMatrix(FEAT_DIM+CLASS_NUM, 6, np.array(target_latent.reshape(1, -1).cpu()), 1)
            if random_A is not None:
                break
        # random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)
        
        # initialize the conditional part
        init_z_class = np.zeros(CLASS_NUM)
        init_z_noise = np.random.normal(loc=0.0, scale=1.0, size=(FEAT_DIM))
        init_z = np.append(init_z_noise, init_z_class)
        init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T
        init_z = np.matmul(random_A, init_low_z)

        self.optimizer = JacobianOptimizer.JacobianOptimizer(FEAT_DIM+CLASS_NUM, 48*320, 
                      lambda zs: myFunc(self.decoder, zs), 
                      lambda xs: myGoodness(target_data, xs), 
                      slider_length, 
                      lambda z: myJacobian(self.decoder, z), 
                      maximizer=False)

        self.optimizer.init(init_z)
        self.best_score = self.optimizer.current_score

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

    def playRealVib(self):
        data, fs = librosa.load('VibTac-12/Texture3.wav', sr=8000)
        sd.play(data*0.5, samplerate=8000)
        self.wav_gif.start()

    def saveWavFile(self):
        scipy.io.wavfile.write("Generated.wav", 44100, self.re_wav)

    def updateValues(self, _update_optimizer_flag):
        # 获取滑块的值
        slider_value = self.slider.value()
        t = slider_value / 999

        if _update_optimizer_flag:
            self.optimizer.update(t)
            print('Next')

        z = self.optimizer.get_z(t)

        x = self.optimizer.f(z.reshape(1, -1))[0]
        spec = x.cpu().detach().numpy().reshape(48, 320)
        score = self.optimizer.g(x.reshape(1, -1))[0]

        print(score)

        ex = np.full((1025 - spec.shape[0], spec.shape[1]), -80)#もとの音声の周波数上限を切っているので配列の大きさを合わせるために-80dbで埋めている
        spec = np.append(spec, ex, axis=0)

        spec = librosa.db_to_amplitude(spec)
        re_wav = librosa.griffinlim(spec,n_iter=100, n_fft=2048, hop_length=int(2048 * 0.1), window='hann')
        self.re_wav = np.tile(20 * re_wav, 10)
        sd.play(self.re_wav)

        self.wav_gif.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeatmapWindow()
    window.show()
    sys.exit(app.exec_())
