# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from libs.models.som import SOM as som


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

        SOMs = []
        for n in range(self.n_class):
            SOMs.append(som(self.Datasets[n], self.child_latent_dim, self.child_resolution,
                            self.child_sigma_max, self.child_sigma_min, self.child_tau, self.cZinit))

        empty = np.empty((self.n_class, self.cK * self.Dim))
        som = SOM(empty, self.parent_latent_dim, self.parent_resolution,
                  self.parent_sigma_max, self.parent_sigma_min, self.parent_tau, self.pZinit)

        self.history['cZeta'] = SOMs[0].Zeta
        self.history['pZeta'] = SOM.Zeta

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        for epoch in bar:
            SOMs_mapping = np.zeros((self.n_class, self.cK * self.Dim))

            for n in range(self.n_class):
                if epoch == 0:
                    SOMs[n]._cooperative_process(epoch)
                    SOMs[n]._adaptive_process()
                    SOMs[n]._competitive_process()
                else:
                    SOMs[n].Y = SOM.Y[n].reshape(self.cK, self.Dim)  # copy back
                    SOMs[n]._competitive_process()
                    SOMs[n]._cooperative_process(epoch)
                    SOMs[n]._adaptive_process()
                SOMs_mapping[n] = SOMs[n].Y.reshape(self.cK * self.Dim)

            SOM.X = SOMs_mapping
            SOM._cooperative_process(epoch)
            SOM._adaptive_process()
            SOM._competitive_process()

            for n in range(self.n_class):
                self.history["cZ"][epoch, n] = SOMs[n].Z
                self.history["cY"][epoch, n] = SOMs[n].Y
                self.history["bmu"][epoch, n] = SOMs[n].bmus
            self.history["pZ"][epoch] = SOM.Z
            self.history["pY"][epoch] = SOM.Y.reshape(self.pK, self.cK, self.Dim)
            self.history["bmm"][epoch] = SOM.bmus
