'''
Author: Mingxin Zhang m.zhang@hapis.k.u-tokyo.ac.jp
Date: 2024-01-11 15:24:25
LastEditors: Mingxin Zhang
LastEditTime: 2024-01-11 18:39:53
Copyright (c) 2024 by Mingxin Zhang, All Rights Reserved. 
'''
import sys
import numpy as np
import sounddevice as sd
import os
import librosa
from functools import partial
from PyQt5.QtWidgets import (QApplication, QRadioButton, QMainWindow, QHBoxLayout, QVBoxLayout, 
                             QWidget, QSlider, QPushButton, QLabel)
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui


class CategoryDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        font = QtGui.QFont()
        font.setFamily('Microsoft YaHei')

        self.setWindowTitle('Categories')
        self.setGeometry(600, 350, 700, 130)
        
        real_data_path = 'Generation_Results/Real/'
        
        self.real_file_list = []
        for root, dirs, files in os.walk(real_data_path):
            for name in files:
                self.real_file_list.append(os.path.join(root, name))
        
        class_list = ['G2 StoneTile', 'G3 CeramicTile', 'G4 CherryTree', 'G6 GrassFibers', 'G8 RoughPaper', 'G9 Jeans']
        
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
        self.hide()
    
    def playVib(self, vib):
        sd.play(vib, samplerate=44100)


class Comparsion(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        font = QtGui.QFont()
        font.setFamily('Microsoft YaHei')

        self.setWindowTitle('Evaluation')
        self.setGeometry(600, 350, 400, 150)
        
        real_data_path = 'Generation_Results/Real/'
        fake_data_path = 'Generation_Results/Generated/'
        
        self.real_file_list = []
        for root, dirs, files in os.walk(real_data_path):
            for name in files:
                self.real_file_list.append(os.path.join(root, name))
                
        self.fake_file_list = []
        for root, dirs, files in os.walk(fake_data_path):
            for name in files:
                self.fake_file_list.append(os.path.join(root, name))
                
        self.class_list = ['G2', 'G3', 'G4', 'G6', 'G8', 'G9']
        
        self.task_n = 0
        self.pairing_pattern = 0
        # 0: Real (Fake) vs Fake (Real)
        # 1: Fake vs Fake
        # 2: Real vs Real
        if self.pairing_pattern == 0:
            file_list = self.real_file_list + self.fake_file_list
        if self.pairing_pattern == 1:
            file_list = self.real_file_list
        if self.pairing_pattern == 2:
            file_list = self.fake_file_list
        
        self.class_order = np.arange(len(self.class_list))
        np.random.shuffle(self.class_order)
        
        self.class_to_guess = self.class_list[self.class_order[self.task_n]]
        
        # Initial status: class 0, Real (Fake) vs Fake (Real)
        while True:
            index = np.random.randint(len(file_list))
            if file_list[index].split('\\')[-1][:2] == self.class_to_guess:
                self.vib_to_guess, fs = librosa.load(file_list[index], sr=44100)
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
                # Initial status: Real (Fake) vs Fake (Real)
                index = np.random.randint(len(file_list))
                if file_list[index].split('\\')[-1][:2] == c:
                    vib_c, fs = librosa.load(file_list[index], sr=44100)
                    self.vib_comp_list.append(vib_c)
                    break
                
        vib_layout_1 = QVBoxLayout()
        play_button_1 = QPushButton("Play")
        play_button_1.clicked.connect(partial(self.playVib, self.vib_comp_list[0]))
        vib_layout_1.addWidget(play_button_1, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_1 = QRadioButton(self.class_list[0], self)
        checkButton_1.toggled.connect(self.checkClass)
        vib_layout_1.addWidget(checkButton_1, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_1)
        
        vib_layout_2 = QVBoxLayout()
        play_button_2 = QPushButton("Play")
        play_button_2.clicked.connect(partial(self.playVib, self.vib_comp_list[1]))
        vib_layout_2.addWidget(play_button_2, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_2 = QRadioButton(self.class_list[1], self)
        checkButton_2.toggled.connect(self.checkClass)
        vib_layout_2.addWidget(checkButton_2, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_2)
        
        vib_layout_3 = QVBoxLayout()
        play_button_3 = QPushButton("Play")
        play_button_3.clicked.connect(partial(self.playVib, self.vib_comp_list[2]))
        vib_layout_3.addWidget(play_button_3, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_3 = QRadioButton(self.class_list[2], self)
        checkButton_3.toggled.connect(self.checkClass)
        vib_layout_3.addWidget(checkButton_3, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_3)
        
        vib_layout_4 = QVBoxLayout()
        play_button_4 = QPushButton("Play")
        play_button_4.clicked.connect(partial(self.playVib, self.vib_comp_list[3]))
        vib_layout_4.addWidget(play_button_4, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_4 = QRadioButton(self.class_list[3], self)
        checkButton_4.toggled.connect(self.checkClass)
        vib_layout_4.addWidget(checkButton_4, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_4)
        
        vib_layout_5 = QVBoxLayout()
        play_button_5 = QPushButton("Play")
        play_button_5.clicked.connect(partial(self.playVib, self.vib_comp_list[4]))
        vib_layout_5.addWidget(play_button_5, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_5 = QRadioButton(self.class_list[4], self)
        checkButton_5.toggled.connect(self.checkClass)
        vib_layout_5.addWidget(checkButton_5, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_5)
        
        vib_layout_6 = QVBoxLayout()
        play_button_6 = QPushButton("Play")
        play_button_6.clicked.connect(partial(self.playVib, self.vib_comp_list[5]))
        vib_layout_6.addWidget(play_button_6, 1, Qt.AlignCenter | Qt.AlignCenter)
        checkButton_6 = QRadioButton(self.class_list[5], self)
        checkButton_6.toggled.connect(self.checkClass)
        vib_layout_6.addWidget(checkButton_6, 1, Qt.AlignCenter | Qt.AlignCenter)
        body_layout.addLayout(vib_layout_6)
        
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
        print(self.class_to_guess, self.checked_class)
        
        if self.task_n == len(self.class_list) - 1:
            self.task_n = 0
            self.pairing_pattern += 1
            if self.pairing_pattern > 2:
                return
            
            self.class_order = np.arange(len(self.class_list))
            np.random.shuffle(self.class_order)
            
        else:
            self.task_n += 1
        
        self.class_to_guess = self.class_list[self.class_order[self.task_n]] 
            
        # 0: Real (Fake) vs Fake (Real)
        # 1: Fake vs Fake
        # 2: Real vs Real
        if self.pairing_pattern == 0:
            file_list = self.real_file_list + self.fake_file_list
        if self.pairing_pattern == 1:
            file_list = self.real_file_list
        if self.pairing_pattern == 2:
            file_list = self.fake_file_list
        
        while True:     
            index = np.random.randint(len(file_list))
            if file_list[index].split('\\')[-1][:2] == self.class_to_guess:
                self.vib_to_guess, fs = librosa.load(file_list[index], sr=44100)
                break
            
        self.vib_comp_list = []
        for c in self.class_list:
            while True:
                index = np.random.randint(len(file_list))
                if file_list[index].split('\\')[-1][:2] == c:
                    vib_c, fs = librosa.load(file_list[index], sr=44100)
                    self.vib_comp_list.append(vib_c)
                    break
        
    
    def playVib(self, vib):
        sd.play(vib, samplerate=44100)
        
    def playVib_guess(self):
        sd.play(self.vib_to_guess, samplerate=44100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    init_window = CategoryDisplay()
    init_window.show()
    sys.exit(app.exec_())