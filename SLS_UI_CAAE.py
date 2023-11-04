'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-07-04 01:27:58
LastEditors: Mingxin Zhang
LastEditTime: 2023-11-04 16:43:17
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
from CAAE_14_norm import model
import torch
import pickle
import sys
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
import pySequentialLineSearch
import joblib
from sklearn.decomposition import PCA
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


device = torch.device("mps")
print(f'Selected device: {device}')

FEAT_DIM = 128
CLASS_NUM = 14
    
    
class HeatmapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Heatmap with Slider")
        self.setGeometry(100, 100, 400, 300)

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

        self.optimizer = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=10)

        self.optimizer.set_hyperparams(kernel_signal_var=0.50,
                                kernel_length_scale=0.10,
                                kernel_hyperparams_prior_var=0.10)
        
        self.optimizer.set_gaussian_process_upper_confidence_bound_hyperparam(5.)

        self.pca = joblib.load('PCA_z10.m')

        self.initUI()

    def initUI(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout(main_widget)

        # 创建显示热度图的区域
        layout.addWidget(QLabel('Target Real Spectrogram'), 1, Qt.AlignCenter | Qt.AlignTop)
        self.figure_real, self.ax_real = plt.subplots()
        self.canvas_real = FigureCanvas(self.figure_real)
        layout.addWidget(self.canvas_real)

        layout.addWidget(QLabel('Generated Spectrogram'), 1, Qt.AlignCenter | Qt.AlignTop)
        self.figure_fake, self.ax_fake = plt.subplots()
        self.canvas_fake = FigureCanvas(self.figure_fake)
        layout.addWidget(self.canvas_fake)

        # 创建滑块并设置范围和初始值
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 1000)
        self.slider.setValue(500)
        layout.addWidget(self.slider)

        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda value: self.updateValues(_update_optimizer_flag=True))
        layout.addWidget(next_button)

        # 连接滑块的valueChanged信号到更新热度图的槽函数
        self.slider.valueChanged.connect(lambda value: self.updateValues(_update_optimizer_flag=False))

        # 初始化热度图
        self.updateValues(_update_optimizer_flag=False)

        self.ax_real.clear()
        self.ax_real.imshow(self.target_spec, cmap='viridis')
        self.ax_real.set_xticks([])
        self.ax_real.set_yticks([])
        self.canvas_real.draw()

    def updateValues(self, _update_optimizer_flag):
        slider_position = self.slider.value() / 999.0

        if _update_optimizer_flag:
            self.optimizer.submit_feedback_data(slider_position)
            print('Next')

        # range of real latent space: [-7.24, 6.42]
        # range of real pca_z: [-8.34 18.69]

        z_low = self.optimizer.calc_point_from_slider_position(slider_position)

        z_low_max = 18.69
        z_low_min = -8.34
        z_low = z_low * (z_low_max - z_low_min) + z_low_min

        z = self.pca.inverse_transform(z_low)

        z = torch.unsqueeze(torch.tensor(z).to(torch.float32), 0)
        z = z.to(device)

        spec = self.decoder(z).cpu().detach().numpy().reshape(48, 320)

        # 绘制热度图
        self.ax_fake.clear()
        self.ax_fake.imshow(spec, cmap='viridis')
        self.ax_fake.set_xticks([])
        self.ax_fake.set_yticks([])
        self.canvas_fake.draw()

        ex = np.full((1025 - spec.shape[0], spec.shape[1]), -80)#もとの音声の周波数上限を切っているので配列の大きさを合わせるために-80dbで埋めている
        spec = np.append(spec, ex, axis=0)

        spec = librosa.db_to_amplitude(spec)
        re_wav = librosa.griffinlim(spec,n_iter=100, n_fft=2048, hop_length=int(2048 * 0.1), window='hann')
        sd.play(np.tile(20*re_wav, 10))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeatmapWindow()
    window.show()
    sys.exit(app.exec_())
