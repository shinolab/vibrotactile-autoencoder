<!--
 * @Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
 * @Date: 2023-03-06 03:05:31
 * @LastEditors: Mingxin Zhang
 * @LastEditTime: 2024-08-22 17:58:43
 * Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
-->
# Tactile Texture Exploration and Synthesis System Using Generative Adversarial Network

This repository contains the code for the paper: [TEXasGAN: Tactile Texture Exploration and Synthesis System Using Generative Adversarial Network](https://arxiv.org/abs/2407.11467)

![System structure](https://github.com/shinolab/vibrotactile-autoencoder/blob/main/images/system.jpg?raw=true)
![Network model](https://github.com/shinolab/vibrotactile-autoencoder/blob/main/images/network.jpg?raw=true)

The opendataset used this paper: [LMT Haptic Texture Database](https://zeus.lmt.ei.tum.de/downloads/texture/) (108 surface materials, SoundScans, Movement)

To obtain the preprocessed dataset, run the notebook `preprocess.ipynb`. In this study, we selected 14 classes to build a training dataset.

Run the `TactileCAAE/train.py` to train the model. The dictionary of the trained model parameters are saved as `encoder.pt`, `decoder.pt` and `classifier.pt`. After loading the trained parameters, the model can be used directly without training.

~~After the running of the `vib_autoencoder.ipynb`, the extracted features are also saved as a dictionary `feat_dict.pickle`.~~
