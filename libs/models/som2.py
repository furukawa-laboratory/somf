# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from libs.models.som import SOM


class SOM2:
    def __init__(self, Datasets, params_1st_SOM, params_2nd_SOM, is_save_history=False):
        self.Datasets = Datasets
        self.n = self.Datasets[0]
        self.params_1st_SOM = params_1st_SOM
        self.params_2nd_SOM = params_2nd_SOM
        self.is_save_history = is_save_history
        self.history = {}

    def fit(self, nb_epoch, verbose=True):
        if self.is_save_history:
            self.history['cZ'] = np.zeros((nb_epoch, self.n_class, self.n_sample, self.child_latent_dim))
            self.history['pZ'] = np.zeros((nb_epoch, self.n_class, self.parent_latent_dim))
            self.history['cY'] = np.zeros((nb_epoch, self.n_class, self.cK, self.Dim))
            self.history['pY'] = np.zeros((nb_epoch, self.pK, self.cK, self.Dim))
            self.history["bmu"] = np.zeros((nb_epoch, self.n_class, self.n_sample))
            self.history["bmm"] = np.zeros((nb_epoch, self.n_class))

        self.1st_soms = []
        for n in range(self.n_class):
            self.1st_soms.append(SOM(self.Datasets[n], **params_1st_SOM))
        self.1st_soms_mapping = np.zeros((self.n_class, self.cK * self.Dim))

        empty = np.empty((self.n_class, self.cK * self.Dim))
        self.2nd_som = SOM(empty, **params_2nd_SOM)

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)
                
        for epoch in bar:
            self._fit_1st_SOMs(epoch, verbose)
            self._fit_2nd_SOM(epoch)

            if self.is_save_history:
                for n, som in enumerate(self.1st_soms):
                    self.history["cZ"][epoch, n] = som.Z
                    self.history["cY"][epoch, n] = som.Y
                    self.history["bmu"][epoch, n] = som.bmus
                self.history["pZ"][epoch] = self.2nd_som.Z
                self.history["pY"][epoch] = self.2nd_som.Y.reshape(self.pK, self.cK, self.Dim)
                self.history["bmm"][epoch] = self.2nd_som.bmus

    def _fit_1st_SOMs(epoch, verbose):
        for n, som in range(self.1st_soms):
            if epoch == 0:
                som._cooperative_process(epoch)
                som._adaptive_process()
                som._competitive_process()
            else:
                som.Y = som.Y[n].reshape(self.cK, self.Dim)  # copy back
                som._competitive_process()
                som._cooperative_process(epoch)
                som._adaptive_process()
            self.1st_soms_mapping[n] = self.2nd_som.Y.reshape(self.cK * self.Dim)
    
    def _fit_2nd_SOM(epoch):
        self.2nd_som.X = self.1st_soms_mapping
        self.2nd_som._cooperative_process(epoch)
        self.2nd_som._adaptive_process()
        self.2nd_som._competitive_process()
