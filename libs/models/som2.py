# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance as dist
from tqdm import tqdm
from sklearn.decomposition import PCA

import libs.models.som as SOM


class SOM2(object):
    def __init__(self, X, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, init='random', metric="sqeuclidean"):
        self.X = X
        self.I = self.X.shape[0]
        self.N = self.X.shape[1]
        self.D = self.X.shape[2]
        self.K = parent_latent_dim
        self.L = child_latent_dim
        self.parent_sigma_max = parent_sigma_max
        self.child_sigma_max = child_sigma_max
        self.parent_sigma_min = parent_sigma_min
        self.child_sigma_min = child_sigma_min
        self.parent_tau = parent_tau
        self.child_tau = child_tau

        self.child_soms = [SOM(X[i], self.L, child_resolution, child_sigma_max, child_sigma_min, child_tau, init) for i in range(self.I)]
        self.parent_som = SOM(X, self.K, parent_resolution, parent_sigma_max, parent_sigma_min, parent_tau, init)

        self.W = np.zeros((self.I, self.L*self.D))

        self.history = {}

    def fit(self, nb_epoch=100):
        self.history['z'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y'] = np.zeros((nb_epoch, self.K, self.D))
        self.history['sigma'] = np.zeros(nb_epoch)

        for _ in nb_epoch:
            # 子SOMからクラスSOMへの逆コピー
            bmus = self.parent_som.bmus
            V = self.parent_som.history['y'][0].reshape(-1, self.L, self.D)[bmus, :, :]

            # 子SOMの更新
            for i, child_som in enumerate(self.child_soms):
                child_som.fit(nb_epoch=1)
                self.W[i] = child_som.history['y'].ravel()

            # 親SOMの更新
            self.parent_som.X = self.W
            self.parent_som.fit(nb_epoch=1)