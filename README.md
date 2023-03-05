<!--
 * @Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
 * @Date: 2023-03-06 03:05:31
 * @LastEditors: Mingxin Zhang
 * @LastEditTime: 2023-03-06 03:23:00
 * Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
-->
# vibtactile-autoencoder

To obtain the preprocessed dataset, run the notebook `preprocess.ipynb`, then the `trainset.pickle` will be generated.

Run the `vib_autoencoder.ipynb` to train the model. The dictionary of the trained model parameters are saved as `encoder.pt`, `decoder.pt` and `classifier.pt`. After loading the trained parameters, the model can be used directly without training.

After the running of the `vib_autoencoder.ipynb`, the extracted features are also saved as a dictionary `feat_dict.pickle`.

