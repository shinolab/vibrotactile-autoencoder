<!--
 * @Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
 * @Date: 2023-03-06 03:05:31
 * @LastEditors: Mingxin Zhang
 * @LastEditTime: 2024-12-24 19:54:44
 * Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
-->
# TexSenseGAN: A User-Guided System for Optimizing Texture-Related Vibrotactile Feedback Using Generative Adversarial Network

This repository contains the code for the paper: [TexSenseGAN: A User-Guided System for Optimizing Texture-Related Vibrotactile Feedback Using Generative Adversarial Network](https://arxiv.org/abs/2407.11467)

![System structure](https://github.com/shinolab/vibrotactile-autoencoder/blob/main/images/system.jpg?raw=true)
![Network model](https://github.com/shinolab/vibrotactile-autoencoder/blob/main/images/network.jpg?raw=true)

The opendataset used this paper: [LMT Haptic Texture Database](https://zeus.lmt.ei.tum.de/downloads/texture/) (108 surface materials, SoundScans, Movement)

To obtain the preprocessed dataset, run the notebook `preprocess.ipynb`. In this study, we selected 14 classes to build a training dataset.

Run the `TactileCAAE/train.py` to train the model. The dictionary of the trained model parameters are saved in `TactileCAAE`. After loading the trained parameters, the model can be used directly for the user optimization.

Run the `DSS_Experiment_UserInitialization.py` to start the optimization with the user initialization. Run the `DSS_Experiment.py` to start the optimization directly.

## Citation

If you find this repo is helpful, please cite:

```bibtex
@article{zhang2024texsensegan,
  title={TexSenseGAN: A User-Guided System for Optimizing Texture-Related Vibrotactile Feedback Using Generative Adversarial Network},
  author={Zhang, Mingxin and Terui, Shun and Makino, Yasutoshi and Shinoda, Hiroyuki},
  journal={arXiv preprint arXiv:2407.11467},
  year={2024}
}
```

## Acknowledgments

This code is based on the implementations of [Difference-Subspace-Search](https://github.com/tbcey74123/Difference-Subspace-Search).
