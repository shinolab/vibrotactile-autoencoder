'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2023-07-04 01:27:58
LastEditors: Mingxin Zhang
LastEditTime: 2023-11-16 16:04:50
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
from GlobalOptimizer import JacobianOptimizer
from CAAE_14class import model
import torch
import pickle
import sys
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
import torchaudio
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


device = torch.device("cuda")
print(f'Selected device: {device}')

FEAT_DIM = 128

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
    xs = torch.tensor(xs).to(device)
    return np.sum((xs.reshape(xs.shape[0], -1) - target.reshape(1, -1)).cpu().detach().numpy() ** 2, axis=1) ** 0.5

def myJacobian(model, z):
    z = z_denormalize(z)
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

        self.setWindowTitle("Heatmap with Slider")
        self.setGeometry(100, 100, 400, 300)

        model_name = 'CAAE_14class'
        self.decoder = model.Generator(feat_dim=FEAT_DIM)
        self.decoder.eval() 
        self.decoder.to(device)

        # Model initialization and parameter loading
        decoder_dict = torch.load(model_name + '/generator_' + str(FEAT_DIM) + 'd.pt', map_location=torch.device('cuda'))
        decoder_dict = {k: v for k, v in decoder_dict.items()}
        self.decoder.load_state_dict(decoder_dict)

        self.griffinlim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=80, hop_length=int(2048 * 0.1), power=1.0)
        self.griffinlim = self.griffinlim.to(device)

        with open('trainset_7-class.pickle', 'rb') as file:
            trainset = pickle.load(file)
    
        index = np.random.randint(len(trainset['spectrogram']))
        self.target_spec = trainset['spectrogram'][index]

        target_data = torch.unsqueeze(torch.tensor(self.target_spec), 0).to(torch.float32).to(device)

        slider_length = getSliderLength(FEAT_DIM, 1, 0.5)
        target_latent = np.random.uniform(low=-2.5, high=2.5, size=(FEAT_DIM))
        # target_data = decoder(target_latent.reshape(1, -1))[0]

        while True:
            random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)
            if random_A is not None:
                break
        # random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)

        init_z = np.random.uniform(low=-2.5, high=2.5, size=(FEAT_DIM))
        # init_z = np.random.normal(loc=0.0, scale=0.5, size=(FEAT_DIM))
        init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T
        init_z = np.matmul(random_A, init_low_z)

        self.optimizer = JacobianOptimizer.JacobianOptimizer(FEAT_DIM, 48*320, 
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
        self.slider.setRange(1, 100)
        self.slider.setValue(50)
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
        # 获取滑块的值
        slider_value = self.slider.value()
        t = slider_value / 99

        if _update_optimizer_flag:
            self.optimizer.update(t)
            print('Next')

        z = self.optimizer.get_z(t)

        print(z.min())
        print(z.max())
        print(z.mean())
        print(z.std())

        x = self.optimizer.f(z.reshape(1, -1))[0]
        spec = x.cpu().detach().numpy().reshape(48, 320)
        score = self.optimizer.g(x.reshape(1, -1))[0]

        print('mean: ', z.mean(), ' std: ', z.std())

        print(score)

        # 绘制热度图
        self.ax_fake.clear()
        self.ax_fake.imshow(spec, cmap='viridis')
        self.ax_fake.set_xticks([])
        self.ax_fake.set_yticks([])
        self.canvas_fake.draw()

        ex = np.full((1025 - spec.shape[0], spec.shape[1]), -80)#もとの音声の周波数上限を切っているので配列の大きさを合わせるために-80dbで埋めている
        spec = np.append(spec, ex, axis=0)

        spec = librosa.db_to_amplitude(spec)
        re_wav = self.griffinlim(torch.tensor(spec).to(device)).cpu().detach().numpy()
        sd.play(np.tile(20*re_wav, 10))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeatmapWindow()
    window.show()
    sys.exit(app.exec_())
