# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

import libs.models.som.SOM as som


class SOM2:
    def __init__(self, Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, init='random'):

        self.Datasets = Datasets
        self.n_class = self.Datasets.shape[0]
        self.n_sample = self.Datasets.shape[1]
        self.Dim = self.Datasets.shape[2]
        self.W = np.zeros((self.n_class, self.cK * self.Dim))
        self.parent_latent_dim = parent_latent_dim
        self.child_latent_dim = child_latent_dim
        self.parent_sigma_max = parent_sigma_max
        self.child_sigma_max = child_sigma_max
        self.parent_sigma_min = parent_sigma_min
        self.child_sigma_min = child_sigma_min
        self.parent_tau = parent_tau
        self.child_tau = child_tau
        self.init = init
        self.pK = parent_resolution ** parent_latent_dim
        self.cK = child_resolution ** child_latent_dim
        self.history = {}

        self._done_fit = False
        self.Z_grad = None

    def fit(self, nb_epoch, verbose=True):

        self.history['cZ'] = np.zeros((nb_epoch, self.n_class, self.n_sample, self.child_latent_dim))
        self.history['pZ'] = np.zeros((nb_epoch, self.n_class, self.n_sample, self.parent_latent_dim))
        self.history['cY'] = np.zeros((nb_epoch, self.n_class, self.cK, self.Dim))
        self.history['pY'] = np.zeros((nb_epoch, self.pY, self.cK, self.Dim))

        SOMs = []
        for n in range(self.n_class):
            SOMs.append(som(self.Datasets[n], self.child_latent_dim,self.child_resolution,
                            self.child_sigma_max, self.child_sigma_min, self.child_tau, self.init))

        empty = np.empty(self.n_class, self.cK * self.Dim)
        SOM = som(empty, self.parent_latent_dim,self.parent_resolution,
                  self.parent_sigma_max, self.parent_sigma_min, self.parent_tau, self.init)

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        for epoch in bar:
            SOMs_manifold = np.zeros((self.n_class, self.cK * self.Dim))

            for n in range(self.n_class):
                if epoch == 0:
                    SOMs[n]._estimate_mapping()
                    SOMs[n]._estimate_Z()
                else:
                    SOMs[n].y_k = SOM.X[n].reshape(self.cK, self.Dim)   # copy back
                    SOMs[n]._estimate_Z()
                    SOMs[n]._estimate_mapping()

                SOMs_manifold[n] = SOMs[n].y.reshape(self.cK * self.Dim)

            SOM._set_learning_data(SOMs_manifold)
            SOM._estimate_mapping()
            SOM._estimate_Z()

            for n in range(self.n_class):
                self.history["cZ"][epoch, n] = SOMs[n].Z
                self.history["cY"][epoch, n] = SOMs[n].y
            self.history["pZ"][epoch] = SOM.Z
            self.history["pY"][epoch] = SOM.y.reshape(self.pK, self.cK, self.Dim)
