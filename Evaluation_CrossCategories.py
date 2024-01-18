'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2024-01-11 15:24:25
LastEditors: Mingxin Zhang
LastEditTime: 2024-01-18 16:02:36
Copyright (c) 2024 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
import sounddevice as sd
import os
import librosa
import datetime
import pandas as pd
from functools import partial
from PyQt5.QtWidgets import (QApplication, QRadioButton, QHBoxLayout, QVBoxLayout, 
                             QWidget, QPushButton, QLabel)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui


class CategoryDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        font = QtGui.QFont()
        font.setFamily('Microsoft YaHei')

        self.setWindowTitle('Categories')
        self.setGeometry(600, 350, 700, 130)
        
        real_data_path = 'Reference_Waves'
        
        self.real_file_list = []
        for root, dirs, files in os.walk(real_data_path):
            for name in files:
                self.real_file_list.append(os.path.join(root, name))
        
        class_list = ['G2 StoneTile', 'G3 CeramicTile', 'G4 CherryTree', 'G6 GrassFibers', 'G8 RoughPaper']
        
        layout = QVBoxLayout()
        
        body_layout = QHBoxLayout()
        
        for c in class_list:
            while True:
                index = np.random.randint(len(self.real_file_list))
                if self.real_file_list[index].split('\\')[-1][:2] == c[:2]:
                    vib_c, fs = librosa.load(self.real_file_list[index], sr=44100)
                    break
                
            vib_layout = QVBoxLayout()
            vib_layout.addWidget(QLabel(c), 1, Qt.AlignCenter | Qt.AlignCenter)
            
            play_button = QPushButton("Play")
            play_button.clicked.connect(partial(self.playVib, vib_c))
            vib_layout.addWidget(play_button, 1, Qt.AlignCenter | Qt.AlignCenter)
            
            body_layout.addLayout(vib_layout)
        
        layout.addLayout(body_layout)
        
        self.submit_button = QPushButton("OK! Enter the experiment")
        self.submit_button.clicked.connect(self.submitRank)
        layout.addWidget(self.submit_button, 1, Qt.AlignCenter | Qt.AlignCenter)
        
        self.setLayout(layout)

    def submitRank(self):
        sd.stop()
        self.new_window = Comparsion()
        self.new_window.show()
        self.close()
    
    def playVib(self, vib):
        sd.play(vib, samplerate=44100)


class Comparsion(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.file_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
 
    def initUI(self):
        font = QtGui.QFont()
        font.setFamily('Microsoft YaHei')

        self.setWindowTitle('Evaluation')
        self.setGeometry(600, 350, 400, 150)
        
        real_data_path = 'Reference_Waves/'
        fake_data_path = 'Generated_Waves/'
        
        self.real_file_list = []
        for root, dirs, files in os.walk(real_data_path):
            for name in files:
                self.real_file_list.append(os.path.join(root, name))
                
        self.fake_file_list = []
        for root, dirs, files in os.walk(fake_data_path):
            for name in files:
                self.fake_file_list.append(os.path.join(root, name))
                
        self.class_list = ['G2', 'G3', 'G4', 'G6', 'G8']
        self.confusion_matrix = {'G2_True':{'G2_User': 0, 'G3_User': 0, 'G4_User': 0, 'G6_User': 0, 'G8_User': 0},
                                 'G3_True':{'G2_User': 0, 'G3_User': 0, 'G4_User': 0, 'G6_User': 0, 'G8_User': 0},
                                 'G4_True':{'G2_User': 0, 'G3_User': 0, 'G4_User': 0, 'G6_User': 0, 'G8_User': 0},
                                 'G6_True':{'G2_User': 0, 'G3_User': 0, 'G4_User': 0, 'G6_User': 0, 'G8_User': 0},
                                 'G8_True':{'G2_User': 0, 'G3_User': 0, 'G4_User': 0, 'G6_User': 0, 'G8_User': 0}}
        self.confusion_matrix = pd.DataFrame(self.confusion_matrix)
        
        self.task_n = 0
        self.repeat_time = 0
        # Only Fake (generated to classify) vs Real (class reference)
        
        self.class_order = np.arange(len(self.class_list))
        np.random.shuffle(self.class_order)
        
        self.class_to_guess = self.class_list[self.class_order[self.task_n]]
        
        # Initial status: class 0
        while True:
            index = np.random.randint(len(self.fake_file_list))
            if self.fake_file_list[index].split('\\')[-1][:2] == self.class_to_guess:
                self.vib_to_guess, fs = librosa.load(self.fake_file_list[index], sr=44100)
                break
        
        layout = QVBoxLayout()
        
        play_button_c = QPushButton("Play the vibration to classify")
        play_button_c.clicked.connect(self.playVib_guess)
        layout.addWidget(play_button_c, 1, Qt.AlignCenter | Qt.AlignCenter)
        
        layout.addWidget(QLabel('Choose the best matching vibration from follows'), 1, Qt.AlignCenter | Qt.AlignCenter)
        
        body_layout = QHBoxLayout()
        
        self.vib_comp_list = []
        
        for c in self.class_list:
            while True:
                index = np.random.randint(len(self.real_file_list))
                if self.real_file_list[index].split('/')[-1][:2] == c:
                    vib_c, fs = librosa.load(self.real_file_list[index], sr=44100)
                    self.vib_comp_list.append(vib_c)
                    break
                
        vib_layout_1 = QVBoxLayout()
        play_button_1 = QPushButton("Play")
        play_button_1.clicked.connect(partial(self.playVib, 0))
        vib_layout_1.addWidget(play_button_1, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_1 = QRadioButton(self.class_list[0], self)
        checkButton_1.toggled.connect(self.checkClass)
        vib_layout_1.addWidget(checkButton_1, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_1)
        
        vib_layout_2 = QVBoxLayout()
        play_button_2 = QPushButton("Play")
        play_button_2.clicked.connect(partial(self.playVib, 1))
        vib_layout_2.addWidget(play_button_2, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_2 = QRadioButton(self.class_list[1], self)
        checkButton_2.toggled.connect(self.checkClass)
        vib_layout_2.addWidget(checkButton_2, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_2)
        
        vib_layout_3 = QVBoxLayout()
        play_button_3 = QPushButton("Play")
        play_button_3.clicked.connect(partial(self.playVib, 2))
        vib_layout_3.addWidget(play_button_3, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_3 = QRadioButton(self.class_list[2], self)
        checkButton_3.toggled.connect(self.checkClass)
        vib_layout_3.addWidget(checkButton_3, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_3)
        
        vib_layout_4 = QVBoxLayout()
        play_button_4 = QPushButton("Play")
        play_button_4.clicked.connect(partial(self.playVib, 3))
        vib_layout_4.addWidget(play_button_4, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_4 = QRadioButton(self.class_list[3], self)
        checkButton_4.toggled.connect(self.checkClass)
        vib_layout_4.addWidget(checkButton_4, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_4)
        
        vib_layout_5 = QVBoxLayout()
        play_button_5 = QPushButton("Play")
        play_button_5.clicked.connect(partial(self.playVib, 4))
        vib_layout_5.addWidget(play_button_5, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_5 = QRadioButton(self.class_list[4], self)
        checkButton_5.toggled.connect(self.checkClass)
        vib_layout_5.addWidget(checkButton_5, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_5)
        
        layout.addLayout(body_layout)
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.setEnabled(False)
        self.submit_button.clicked.connect(self.submitClass)
        layout.addWidget(self.submit_button, 1, Qt.AlignCenter | Qt.AlignCenter)
        
        self.setLayout(layout)

    def checkClass(self):
        self.checkButton = self.sender()
        # self.lMessage.setText(u'Choose {0}'.format(checkButton.text()))
        self.checked_class = self.checkButton.text()
        self.submit_button.setEnabled(True)

    def submitClass(self):
        self.checkButton.setCheckable(False)
        self.checkButton.setCheckable(True)
        sd.stop()
        self.submit_button.setEnabled(False)
        self.confusion_matrix[self.class_to_guess + '_True'][self.checked_class + '_User'] += 1
        # print(self.confusion_matrix)
        self.confusion_matrix.to_csv('Evaluation_Results/confusion_matrix' + self.file_time + '.csv')
        
        if self.task_n == len(self.class_list) - 1:
            self.task_n = 0
            self.repeat_time += 1
            # 3 repetition experiments
            if self.repeat_time > 2:
                sys.exit()
            
            self.class_order = np.arange(len(self.class_list))
            np.random.shuffle(self.class_order)
            
        else:
            self.task_n += 1
        
        self.class_to_guess = self.class_list[self.class_order[self.task_n]] 
            
        # generated vibration to guess (Fake)
        while True:     
            index = np.random.randint(len(self.fake_file_list))
            if self.fake_file_list[index].split('\\')[-1][:2] == self.class_to_guess:
                self.vib_to_guess, fs = librosa.load(self.fake_file_list[index], sr=44100)
                break
            
        # reference vibrations
        self.vib_comp_list = []
        for c in self.class_list:
            while True:
                index = np.random.randint(len(self.real_file_list))
                if self.real_file_list[index].split('/')[-1][:2] == c:
                    vib_c, fs = librosa.load(self.real_file_list[index], sr=44100)
                    self.vib_comp_list.append(vib_c)
                    break
        
    
    def playVib(self, vib_index):
        sd.play(self.vib_comp_list[vib_index], samplerate=44100)
        
    def playVib_guess(self):
        sd.play(self.vib_to_guess, samplerate=44100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    init_window = CategoryDisplay()
    init_window.show()
    sys.exit(app.exec_())