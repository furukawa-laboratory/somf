# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from libs.models.som import SOM


class SOM2:
    def __init__(self, datasets, params_first_som, params_second_som, is_save_history=False):
        self.datasets = datasets
        self.params_first_som = params_first_som
        self.params_second_som = params_second_som
        self.is_save_history = is_save_history
        self.n_class = self.datasets.shape[0]
        self.n_sample = datasets.shape[1]   # 1クラスごとのサンプル数は一定と仮定（要修正）
        self.Dim = self.datasets.shape[2]
        self.cK = params_first_som["resolution"] ** params_first_som["latent_dim"]
        self.pK = params_second_som["resolution"] ** params_second_som["latent_dim"]
        self.first_latent_dim = params_first_som["latent_dim"]
        self.second_latent_dim = params_second_som["latent_dim"]
        self.history = {}

    def fit(self, nb_epoch, verbose=True):
        if self.is_save_history:
            self.history['cZ'] = np.zeros((nb_epoch, self.n_class, self.n_sample, self.first_latent_dim))
            self.history['pZ'] = np.zeros((nb_epoch, self.n_class, self.second_latent_dim))
            self.history['pZeta'] = np.zeros((self.pK, self.second_latent_dim))
            self.history['cY'] = np.zeros((nb_epoch, self.n_class, self.cK, self.Dim))
            self.history['pY'] = np.zeros((nb_epoch, self.pK, self.cK, self.Dim))
            self.history["bmu"] = np.zeros((nb_epoch, self.n_class, self.n_sample))
            self.history["bmm"] = np.zeros((nb_epoch, self.n_class))

        self.first_soms = []
        for n in range(self.n_class):
            self.first_soms.append(SOM(self.datasets[n], **self.params_first_som))
        self.first_soms_mapping = np.zeros((self.n_class, self.cK * self.Dim))

        dummy_data = np.empty((self.n_class, self.cK * self.Dim))
        self.second_som = SOM(dummy_data, **self.params_second_som)
        self.history["pZeta"] = self.second_som.Zeta

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)
                
        for epoch in bar:
            self._fit_first_SOMs(epoch)
            self._fit_second_SOM(epoch)

            if self.is_save_history:
                for n, som in enumerate(self.first_soms):
                    self.history["cZ"][epoch, n] = som.Z
                    self.history["cY"][epoch, n] = som.Y
                    self.history["bmu"][epoch, n] = som.bmus
                self.history["pZ"][epoch] = self.second_som.Z
                self.history["pY"][epoch] = self.second_som.Y.reshape(self.pK, self.cK, self.Dim)
                self.history["bmm"][epoch] = self.second_som.bmus

    def _fit_first_SOMs(self, epoch):
        for n, som in enumerate(self.first_soms):
            if epoch == 0:
                som._cooperative_process(epoch)
                som._adaptive_process()
                som._competitive_process()
            else:
                som.Y = self.second_som.Y[n].reshape(self.cK, self.Dim)  # copy back
                som._competitive_process()
                som._cooperative_process(epoch)
                som._adaptive_process()
            self.first_soms_mapping[n] = som.Y.reshape(self.cK * self.Dim)
    
    def _fit_second_SOM(self, epoch):
        self.second_som.X = self.first_soms_mapping
        self.second_som._cooperative_process(epoch)
        self.second_som._adaptive_process()
        self.second_som._competitive_process()
