'''
Author: Mingxin Zhang m.zhang@hapis.u-tokyo.ac.jp
Date: 2023-04-12 01:47:50
LastEditors: Mingxin Zhang
LastEditTime: 2023-07-01 21:13:22
Copyright (c) 2023 by Mingxin Zhang, All Rights Reserved. 
'''

import pySequentialLineSearch
from sklearn.decomposition import PCA
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

FEAT_DIM = 512

class GlobalOptimizer():
    def __init__(self, n, m, f, g, search_range, maximizer=True):
        self.n = n
        self.m = m
        self.f = f
        self.g = g
        self.search_range = search_range
        self.maximizer = maximizer

    def init(self, init_z):
        self.current_z = init_z
        self.current_x = self.f(self.current_z.reshape(1, -1))[0]
        self.current_score = self.g(self.current_x.reshape(1, -1))

        print('Initialize', self.name, 'with score', self.current_score )

    def get_z(self, t):
        pass

    def find_optimal(self, sample_n, batch_size=100):
        if batch_size <= 0:
            batch_size = sample_n

        zs = []
        for i in range(sample_n):
            t = i / float(sample_n - 1)
            z = self.get_z(t)
            zs.append(z)
        zs = np.array(zs)

        batch_n = sample_n // batch_size
        remainder = sample_n - batch_size * batch_n

        xs = np.zeros((sample_n, self.m))
        for i in range(batch_n):
            xs[i * batch_size:(i + 1) * batch_size] = self.f(zs[i * batch_size:(i + 1) * batch_size])
        if remainder != 0:
            xs[batch_n * batch_size:] = self.f(zs[batch_n * batch_size:])

        scores = self.g(xs)

        if self.maximizer:
            idx = np.argmax(scores)
        else:
            idx = np.argmin(scores)
        t = idx / float(sample_n - 1)

        z = self.get_z(t)
        x = self.f(z.reshape(1, -1))[0]
        score = self.g(x.reshape(1, -1))[0]

        return z, x, score, t

    def update(self, t):
        pass


class JacobianOptimizer(GlobalOptimizer):
    def __init__(self, n, m, f, g, search_range, jacobian_func, maximizer=True):
        super(JacobianOptimizer, self).__init__(n, m, f, g, search_range, maximizer=maximizer)
        self.name = 'Jacobian Optimizer'
        self.jacobian_func = jacobian_func
        self.center_tolerance = 0.05

    def init(self, init_z):
        super(JacobianOptimizer, self).init(init_z)

        self.update_jacobian(self.current_z)
        self.sample_direction()

    def get_z(self, t):
        return self.current_z + self.subspace_basis * (t - 0.5) * self.search_range * 2

    def update_jacobian(self, z):
        self.jacobian = self.jacobian_func(z)
        u, s, vh = np.linalg.svd(self.jacobian, full_matrices=True)

        num = s.shape[0]
        self.jacobian_vhs = vh[:num]
        self.jacobian_s = s[:num]
        self.jacobian_mask = np.ones(self.jacobian_vhs.shape[0], dtype=bool)

    def sample_direction(self):
        if self.jacobian_s[self.jacobian_mask].shape[0] <= 0:
            self.jacobian_mask = np.ones(self.jacobian_vhs.shape[0], dtype=bool)

        s = self.jacobian_s[self.jacobian_mask] + 1e-6
        vh = self.jacobian_vhs[self.jacobian_mask]

        # Importance sampling
        choice_p = s / np.sum(s)
        idx = np.random.choice(choice_p.shape[0], p=choice_p)

        self.subspace_basis = vh[idx]

        self.jacobian_mask[np.arange(self.jacobian_vhs.shape[0])[self.jacobian_mask][idx]] = False

    def update(self, t):
        self.current_z = self.get_z(t)
        self.current_x = self.f(self.current_z.reshape(1, -1))[0]
        self.current_score = self.g(self.current_x.reshape(1, -1))[0]

        if np.abs(t - 0.5) > self.center_tolerance:
            self.update_jacobian(self.current_z)
        self.sample_direction()



def calc_model_gradient(model, latent_vector):
        jacobian = calc_model_gradient_FDM(model, latent_vector, delta=1e-2)
        return jacobian

def calc_model_gradient_FDM(model, latent_vector, delta=1e-4):
    sample_latents = np.repeat(latent_vector.reshape(1, -1), repeats=FEAT_DIM + 1, axis=0)
    sample_latents[1:] += np.identity(FEAT_DIM) * delta

    sample_datas = model.inverse_transform(sample_latents)
    sample_datas = sample_datas.reshape(-1, 12*100)

    jacobian = (sample_datas[1:] - sample_datas[0]).T / delta
    return jacobian

def myFunc(pca, zs):
    output = pca.inverse_transform(zs).reshape(zs.shape[0], -1)
    # output = decoder(zs).reshape(zs.shape[0], -1)
    return output

def myGoodness(target, xs):
    return np.sum((xs.reshape(xs.shape[0], -1) - target.reshape(1, -1)) ** 2, axis=1) ** 0.5

def myJacobian(model, z):
    return calc_model_gradient(model, z)

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

def main():
    with open('trainset.pickle', 'rb') as file:
        data = pickle.load(file)

    X = data['spectrogram']
    X = X.reshape(*X.shape[:-2], -1)
    print(X.shape)

    pca = PCA(n_components=FEAT_DIM) 
    X_reduced = pca.fit_transform(X)

    with open('sample_target_spec_2.pickle', 'rb') as file:
        target_spec = pickle.load(file)
    # plt.imshow(target_spec)
    # plt.show()
    # target_spec = np.expand_dims(target_spec, axis=0)

    slider_length = getSliderLength(FEAT_DIM, 1, 0.2)
    target_latent = np.random.uniform(-1, 1, FEAT_DIM)
    # target_data = decoder(target_latent.reshape(1, -1))[0]

    while True:
        random_A = getRandomAMatrix(FEAT_DIM, 6, np.array(target_latent.reshape(1, -1)), 1)
        if random_A is not None:
            break
    # random_A = getRandomAMatrix(FEAT_DIM, 6, target_latent.reshape(1, -1), 1)
    
    init_z = np.random.uniform(low=-1, high=1, size=(FEAT_DIM))
    init_low_z = np.matmul(np.linalg.pinv(random_A), init_z.T).T
    init_z = np.matmul(random_A, init_low_z)

    print(slider_length)

    optimizer = JacobianOptimizer(FEAT_DIM, 12*100, 
                      lambda zs: myFunc(pca, zs), 
                      lambda xs: myGoodness(target_spec, xs), 
                      slider_length, 
                      lambda z: myJacobian(pca, z), 
                      maximizer=False)

    optimizer.init(init_z)
    best_score = optimizer.current_score

    iter_num = 1000
    
    for i in range(iter_num):
        n_sample = 1000
        opt_z, opt_x, opt_score, opt_t = optimizer.find_optimal(n_sample, batch_size=n_sample)
        if opt_score < best_score:
            best_score = opt_score

        print('Iteration #' + str(i) + ': ' + str(best_score))
        optimizer.update(opt_t)

    fig, ax = plt.subplots(2, 1, figsize=(5, 3)) 
    fig.suptitle('Iter num = ' + str(iter_num) + ', loss = ' + str(best_score), fontsize=16)
    ax[0].imshow(target_spec.reshape(12, 100)) 
    ax[0].set_title("Original") 
    ax[1].imshow(opt_x.reshape(12, 100)) 
    ax[1].set_title("Generated")
    plt.show()


if __name__ == '__main__':
    main()

