# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from libs.models.som import SOM


class SOM2:
    def __init__(self, Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZinit, cZinit):

        self.Datasets = Datasets
        self.n_class = self.Datasets.shape[0]
        self.n_sample = self.Datasets.shape[1]
        self.Dim = self.Datasets.shape[2]
        self.parent_latent_dim = parent_latent_dim
        self.child_latent_dim = child_latent_dim
        self.parent_resolution = parent_resolution
        self.child_resolution = child_resolution
        self.parent_sigma_max = parent_sigma_max
        self.child_sigma_max = child_sigma_max
        self.parent_sigma_min = parent_sigma_min
        self.child_sigma_min = child_sigma_min
        self.parent_tau = parent_tau
        self.child_tau = child_tau
        self.pZinit = pZinit
        self.cZinit = cZinit
        self.pK = parent_resolution ** parent_latent_dim
        self.cK = child_resolution ** child_latent_dim
        self.W = np.zeros((self.n_class, self.cK * self.Dim))
        self.history = {}

        self._done_fit = False
        self.Z_grad = None

    def fit(self, nb_epoch, verbose=True):

        self.history['cZ'] = np.zeros((nb_epoch, self.n_class, self.n_sample, self.child_latent_dim))
        self.history['pZ'] = np.zeros((nb_epoch, self.n_class, self.parent_latent_dim))
        self.history['cY'] = np.zeros((nb_epoch, self.n_class, self.cK, self.Dim))
        self.history['pY'] = np.zeros((nb_epoch, self.pK, self.cK, self.Dim))
        self.history["bmu"] = np.zeros((nb_epoch, self.n_class, self.n_sample))
        self.history["bmm"] = np.zeros((nb_epoch, self.n_class))

        soms = []
        for n in range(self.n_class):
            soms.append(SOM(self.Datasets[n], self.child_latent_dim, self.child_resolution,
                            self.child_sigma_max, self.child_sigma_min, self.child_tau, self.cZinit))

        empty = np.empty((self.n_class, self.cK * self.Dim))
        som = SOM(empty, self.parent_latent_dim, self.parent_resolution,
                  self.parent_sigma_max, self.parent_sigma_min, self.parent_tau, self.pZinit)

        self.history['cZeta'] = soms[0].Zeta
        self.history['pZeta'] = som.Zeta

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        for epoch in bar:
            soms_mapping = np.zeros((self.n_class, self.cK * self.Dim))

            for n in range(self.n_class):
                if epoch == 0:
                    soms[n]._cooperative_process(epoch)
                    soms[n]._adaptive_process()
                    soms[n]._competitive_process()
                else:
                    soms[n].Y = som.Y[n].reshape(self.cK, self.Dim)  # copy back
                    soms[n]._competitive_process()
                    soms[n]._cooperative_process(epoch)
                    soms[n]._adaptive_process()
                soms_mapping[n] = soms[n].Y.reshape(self.cK * self.Dim)

            som.X = soms_mapping
            som._cooperative_process(epoch)
            som._adaptive_process()
            som._competitive_process()

            for n in range(self.n_class):
                self.history["cZ"][epoch, n] = soms[n].Z
                self.history["cY"][epoch, n] = soms[n].Y
                self.history["bmu"][epoch, n] = soms[n].bmus
            self.history["pZ"][epoch] = som.Z
            self.history["pY"][epoch] = som.Y.reshape(self.pK, self.cK, self.Dim)
            self.history["bmm"][epoch] = som.bmus
