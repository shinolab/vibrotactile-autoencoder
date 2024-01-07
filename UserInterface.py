import torch
import librosa
import datetime
import scipy
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import Methods
import pickle
from GlobalOptimizer import JacobianOptimizer
from PyQt5.QtWidgets import (QRadioButton, QMainWindow, QHBoxLayout, QVBoxLayout, 
                             QWidget, QSlider, QPushButton, QLabel)
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

device = torch.device("cuda")
print(f'Selected device: {device}')

FEAT_DIM = 128
SLIDER_LEN = 30


class InitWindow(QWidget):
    def __init__(self, griffinlim, target_spec, target_group, decoder, init_z, task):
        super().__init__()

        self.griffinlim = griffinlim
        self.target_spec = target_spec
        self.target_group = target_group
        self.decoder = decoder
        self.init_z = init_z
        self.task = task

        self.initUI()
 
    def initUI(self):
        font = QtGui.QFont()
        font.setFamily('Microsoft YaHei')

        self.setWindowTitle('Initialization')
        self.setGeometry(100, 100, 400, 300)

        real_vib = self.spec2wav(self.target_spec)
        real_vib = real_vib * 100
        self.real_vib = np.tile(real_vib, 10)

        layout = QVBoxLayout()

        real_vib_layout = QHBoxLayout()
        real_vib_layout.addWidget(QLabel('Click to play the target vibration'), 1, Qt.AlignCenter | Qt.AlignCenter)

        play_real_button = QPushButton("Play target vibration")
        play_real_button.clicked.connect(self.playRealVib)
        real_vib_layout.addWidget(play_real_button, 1, Qt.AlignCenter | Qt.AlignCenter)

        layout.addLayout(real_vib_layout)
        layout.addSpacing(18)

        self.init_vib = self.z2wav(self.init_z)

        init_vib_layout = QHBoxLayout()
        init_vib_layout.addWidget(QLabel('Click to play the initial vibration'), 1, Qt.AlignCenter | Qt.AlignCenter)

        play_init_button = QPushButton("Play initial vibration")
        play_init_button.clicked.connect(self.playInitVib)
        init_vib_layout.addWidget(play_init_button, 1, Qt.AlignCenter | Qt.AlignCenter)
        layout.addLayout(init_vib_layout)

        layout.addWidget(QLabel('If the vibration like the target? - Please rank it'), 1, Qt.AlignCenter | Qt.AlignCenter)

        rank_list = ["Good 👍", "So-so👌", "Bad 👎"]

        rank_button_layout = QHBoxLayout()
        for i in range(len(rank_list)):
            checkButton = QRadioButton(rank_list[i], self)
            checkButton.toggled.connect(self.checkRank)
            rank_button_layout.addWidget(checkButton)
        
        layout.addLayout(rank_button_layout)

        self.good_num = 0
        self.soso_num = 0
        self.bad_num = 0
        self.good_list = []
        self.soso_list = []
        self.bad_list = []
        self.tabu_list = []
        self.lMessage = QLabel(u"Good 👍: {0},\tSo-so👌: {1},\tBad 👎: {2}"\
                               .format(self.good_num, self.soso_num, self.bad_num))
        layout.addWidget(self.lMessage)

        self.submit_button = QPushButton("Submit the rank")
        self.submit_button.setEnabled(False)
        self.submit_button.clicked.connect(self.submitRank)
        layout.addWidget(self.submit_button, 1, Qt.AlignCenter | Qt.AlignCenter)
        
        self.setLayout(layout)

    def checkRank(self):
        self.checkButton = self.sender()
        # self.lMessage.setText(u'Choose {0}'.format(checkButton.text()))
        self.rank = self.checkButton.text()
        self.submit_button.setEnabled(True)

    def SelectVec(self, z, rank):
        # with open('CAAE_14class/latent_dict.pickle', 'rb') as file:
        #     latent_dict = pickle.load(file)
        
        # avg_dis = 0
        # dis_num = 0
        # # Calculate the average distance of feature
        # for i in range(len(latent_dict['z'])):
        #      for j in range(i+1, len(latent_dict['z'])):
        #           dis = np.linalg.norm(np.array(latent_dict['z'][i]) - np.array(latent_dict['z'][j]))
        #           avg_dis += dis
        #           dis_num += 1
        
        # avg_dis /= dis_num
        # print(avg_dis)
        avg_dis = 14.095
        step = avg_dis / 8
        tabu_range = step

        with open('CAAE_14class/latent_dict.pickle', 'rb') as file:
            latent_dict = pickle.load(file)

        new_z = []
        # No.0 Good
        if rank == 0:
            while True:
                index = np.random.randint(len(latent_dict['z']))
                new_z = latent_dict['z'][index]
                # print(latent_dict['label'][index])

                dis = np.linalg.norm(np.array(z) - np.array(new_z))
                if dis <= step:
                    break
        # No.1 So-so
        elif rank == 1:
            while True:
                index = np.random.randint(len(latent_dict['z']))
                new_z = latent_dict['z'][index]

                if self.tabu_list != []:
                    if_continue = True
                    while if_continue:
                        if_continue = False
                        for tabu_element in self.tabu_list:
                            dis = np.linalg.norm(np.array(tabu_element) - np.array(new_z))
                            if dis <= tabu_range:
                                if_continue = True
                                index = np.random.randint(len(latent_dict['z']))
                                new_z = latent_dict['z'][index]
                                break

                dis = np.linalg.norm(np.array(z) - np.array(new_z))
                if dis >= step / 2 and dis <= 2 * step:
                    # print(latent_dict['label'][index])
                    break
        # No.2 Bad
        elif rank == 2:
            while True:
                index = np.random.randint(len(latent_dict['z']))
                new_z = latent_dict['z'][index]

                if self.tabu_list != []:
                    if_continue = True
                    while if_continue:
                        if_continue = False
                        for tabu_element in self.tabu_list:
                            dis = np.linalg.norm(np.array(tabu_element) - np.array(new_z))
                            if dis <= tabu_range:
                                if_continue = True
                                index = np.random.randint(len(latent_dict['z']))
                                new_z = latent_dict['z'][index]
                                break

                dis = np.linalg.norm(np.array(z) - np.array(new_z))
                if dis >= 2 * step:
                    # print(latent_dict['label'][index])
                    break

        self.tabu_list.append(new_z)
        if len(self.tabu_list) > 5:
            self.tabu_list.pop(0)
        return new_z

    def submitRank(self):
        if self.rank == "Good 👍":
            self.rank = 0
            self.good_num += 1
            self.good_list.append(self.init_z)
        elif self.rank == "So-so👌":
            self.rank = 1
            self.soso_num += 1
            self.soso_list.append(self.init_z)
        else:
            self.rank = 2
            self.bad_num += 1
        self.lMessage.setText(u"Good 👍: {0},\tSo-so👌: {1},\tBad 👎: {2}"\
                               .format(self.good_num, self.soso_num, self.bad_num))

        self.checkButton.setCheckable(False)
        self.checkButton.setCheckable(True)
        sd.stop()
        self.submit_button.setEnabled(False)

        if self.good_num >= 1:
            good_mean = np.array(self.good_list).mean(axis=0)

            # if self.soso_list != []:
            #     soso_mean = np.array(self.soso_list).mean(axis=0)
            #     init_z = 0.9 * good_mean + 0.1 * soso_mean
            # else:
            #     init_z = good_mean
            
            init_z = good_mean

            target_latent = np.random.uniform(-2.5, 2.5, FEAT_DIM)
            target_latent = torch.tensor(target_latent).to(torch.float32).to(device)

            while True:
                random_A = Methods.getRandomAMatrix(FEAT_DIM, 6, np.array(target_latent.reshape(1, -1).cpu()), 1)
                if random_A is not None:
                    break
            # random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)

            # initialize the latent
            init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T
            init_z = np.matmul(random_A, init_low_z)

            if self.task == 'Visualization':
                self.new_window = DSS_Visualization(self.griffinlim, 
                                                    self.target_spec, 
                                                    self.decoder, 
                                                    init_z)
            if self.task == 'Experiment':
                self.new_window = DSS_Experiment(self.griffinlim, 
                                                 self.target_spec, 
                                                 self.target_group, 
                                                 self.decoder, 
                                                 init_z)
            self.new_window.show()
            self.hide()
            return

        new_z = self.SelectVec(self.init_z, self.rank)
        self.init_vib = self.z2wav(new_z)

    def z2wav(self, z):
        spec = self.decoder(torch.tensor(z).unsqueeze(dim=0).to(torch.float32).to(device))
        spec = Methods.img_denormalize(spec)
        spec = spec.cpu().detach().squeeze().numpy()
        wav = self.spec2wav(spec)
        wav = wav * 100
        wav = np.tile(wav, 10)
        return wav

    def spec2wav(self, spec):
        ex = np.full((1025 - spec.shape[0], spec.shape[1]), -80) #もとの音声の周波数上限を切っているので配列の大きさを合わせるために-80dbで埋めている
        spec = np.append(spec, ex, axis=0)

        spec = librosa.db_to_amplitude(spec)
        re_wav = self.griffinlim(torch.tensor(spec).to(device))

        return re_wav.cpu().detach().numpy()
    
    def playRealVib(self):
        sd.play(self.real_vib, samplerate=44100)

    def playInitVib(self):
        sd.play(self.init_vib, samplerate=44100)
    

class DSS_Visualization(QMainWindow):
    def __init__(self, griffinlim, target_spec, decoder, init_z):
        super().__init__()

        self.setWindowTitle("DSS Visualization")
        self.setGeometry(100, 100, 400, 300)

        self.griffinlim = griffinlim
        self.decoder = decoder

        self.target_spec = target_spec
        target_data = torch.unsqueeze(torch.tensor(self.target_spec), 0).to(torch.float32).to(device)

        slider_length = Methods.getSliderLength(FEAT_DIM, 1, 0.8)

        self.optimizer = JacobianOptimizer.JacobianOptimizer(FEAT_DIM, 48*320, 
                      lambda zs: Methods.myFunc(self.decoder, zs), 
                      lambda xs: Methods.myGoodness(target_data, xs), 
                      slider_length, 
                      lambda z: Methods.myJacobian(self.decoder, z), 
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
        self.slider.setRange(1, int(SLIDER_LEN))
        self.slider.setValue(int(SLIDER_LEN / 2))
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
        t = slider_value / (SLIDER_LEN - 1)

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
        sd.play(np.tile(100*re_wav, 10))


class DSS_Experiment(QMainWindow):
    def __init__(self, griffinlim, target_spec, target_group, decoder, init_z):
        super().__init__()

        self.setWindowTitle("Vibration Optimizer")
        self.setGeometry(100, 100, 200, 400)

        self.griffinlim = griffinlim
        self.target_spec = target_spec
        self.target_group = target_group
        self.decoder = decoder
        self.init_z = init_z

        self.target_wav = self.spec2wav(self.target_spec)
        # meter = pyln.Meter(44100) # create BS.1770 meter
        # self.target_loudness = meter.integrated_loudness(self.target_wav)
        self.target_wav = self.target_wav * 100
        self.target_wav = np.tile(self.target_wav, 10)

        self.target_spec = target_spec
        self.target_data = torch.unsqueeze(torch.tensor(self.target_spec), 0).to(torch.float32).to(device)

        self.slider_length = Methods.getSliderLength(FEAT_DIM, 1, 0.8)

        self.optimizer = JacobianOptimizer.JacobianOptimizer(FEAT_DIM, 48*320, 
                      lambda zs: Methods.myFunc(self.decoder, zs), 
                      lambda xs: Methods.myGoodness(self.target_data, xs), 
                      self.slider_length, 
                      lambda z: Methods.myJacobian(self.decoder, z), 
                      maximizer=False)

        self.optimizer.init(self.init_z)
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

        self.wav_gif = QMovie('UI/ezgif-2-ea9f643ae8.gif')

        self.wav_gif = QMovie()
        self.wav_gif.setFileName('UI/ezgif-2-ea9f643ae8.gif')
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
        self.slider.setRange(1, int(SLIDER_LEN))
        self.slider.setValue(int(SLIDER_LEN / 2))
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

        layout.addWidget(QLabel('Click \'Reset\' to restart the optimization'), 1, Qt.AlignCenter | Qt.AlignBottom)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.restart)
        layout.addWidget(reset_button)

        # 连接滑块的valueChanged信号到更新热度图的槽函数
        self.slider.valueChanged.connect(lambda value: self.updateValues(_update_optimizer_flag=False))

        self.updateValues(_update_optimizer_flag=False)
        sd.stop()
    
    def spec2wav(self, spec):
        ex = np.full((1025 - spec.shape[0], spec.shape[1]), -80) #もとの音声の周波数上限を切っているので配列の大きさを合わせるために-80dbで埋めている
        spec = np.append(spec, ex, axis=0)

        spec = librosa.db_to_amplitude(spec)
        re_wav = self.griffinlim(torch.tensor(spec).to(device))

        return re_wav.cpu().detach().numpy()

    def playRealVib(self):
        # play_wav = pyln.normalize.loudness(self.target_wav, self.target_loudness, NORMALIZED_DB)
        sd.play(self.target_wav, samplerate=44100)
        self.wav_gif.start()

    def saveWavFile(self):
        file_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        subject_name = 'zhang'
        real_file_name = "Generation_Results/Real/" + subject_name + "/" + self.target_group + "_" + file_time + ".wav"
        fake_file_name = "Generation_Results/Generated/" + subject_name + "/" + self.target_group + "_" + file_time + ".wav"
        scipy.io.wavfile.write(real_file_name, 44100, self.target_wav)
        scipy.io.wavfile.write(fake_file_name, 44100, self.re_wav)
    
    def restart(self):
        self.slider.setValue(int(SLIDER_LEN / 2))
        sd.stop()
        self.optimizer = JacobianOptimizer.JacobianOptimizer(FEAT_DIM, 48*320, 
                      lambda zs: Methods.myFunc(self.decoder, zs), 
                      lambda xs: Methods.myGoodness(self.target_data, xs), 
                      self.slider_length, 
                      lambda z: Methods.myJacobian(self.decoder, z), 
                      maximizer=False)

        self.optimizer.init(self.init_z)
        self.best_score = self.optimizer.current_score
        self.wav_gif.stop()

    def updateValues(self, _update_optimizer_flag):
        # 获取滑块的值
        slider_value = self.slider.value()
        t = slider_value / (SLIDER_LEN - 1)

        if _update_optimizer_flag:
            self.optimizer.update(t)
            # print('Next')

        z = self.optimizer.get_z(t)

        x = self.optimizer.f(z.reshape(1, -1))[0]
        spec = x.cpu().detach().numpy().reshape(48, 320)

        re_wav = self.spec2wav(spec)

        # meter = pyln.Meter(44100) # create BS.1770 meter
        # loudness = meter.integrated_loudness(re_wav)

        # loudness normalize audio to target
        # loudness_normalized_audio = pyln.normalize.loudness(re_wav, loudness, NORMALIZED_DB)
        loudness_normalized_audio = re_wav * 100

        self.re_wav = np.tile(loudness_normalized_audio, 10)
        sd.play(self.re_wav)

        self.wav_gif.stop()