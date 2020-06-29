# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from libs.models.som import SOM


class SOM2:
    def __init__(self, datasets, params_first_som, params_second_som, is_save_history=False):
        self.datasets = datasets
        self.n_class = len(datasets)
        self.dim = len(datasets[0][0])

        self.params_first_som = params_first_som
        self.n_first_node = params_first_som["resolution"] ** params_first_som["latent_dim"]
        self.first_latent_dim = params_first_som["latent_dim"]

        self.params_second_som = params_second_som
        self.n_second_node = params_second_som["resolution"] ** params_second_som["latent_dim"]
        self.second_latent_dim = params_second_som["latent_dim"]

        self.is_save_history = is_save_history
        self.history = {}

        assert self.first_latent_dim in [1, 2], "Not Implemented Error. first_latent_dim must be 1 or 2."
        assert self.second_latent_dim in [1, 2], "Not Implemented Error. second_latent_dim must be 1 or 2."


    def fit(self, nb_epoch, verbose=True):
        if self.is_save_history:
            self.history['cZ'] = []
            self.history['pZ'] = np.zeros((nb_epoch, self.n_class, self.second_latent_dim))
            self.history['pZeta'] = np.zeros((self.n_second_node, self.second_latent_dim))
            self.history['cY'] = np.zeros((nb_epoch, self.n_class, self.n_first_node, self.dim))
            self.history['pY'] = np.zeros((nb_epoch, self.n_second_node, self.n_first_node, self.dim))
            self.history['bmu'] = []
            self.history["bmm"] = np.zeros((nb_epoch, self.n_class))

        self.first_soms = []
        for n in range(self.n_class):
            if isinstance(self.datasets[n], list):
                self.first_soms.append(SOM(np.array(self.datasets[n]), **self.params_first_som))
            else:
                self.first_soms.append(SOM(self.datasets[n], **self.params_first_som))
        self.first_soms_mapping = np.zeros((self.n_class, self.n_first_node * self.dim))

        dummy_data = np.empty((self.n_class, self.n_first_node * self.dim))
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
                cZ = []
                bmu = []
                for n, som in enumerate(self.first_soms):
                    cZ.append(som.Z)
                    bmu.append(som.bmus)
                    self.history["cY"][epoch, n] = som.Y
                self.history["cZ"].append(cZ)
                self.history["bmu"].append(bmu)
                self.history["pZ"][epoch] = self.second_som.Z
                self.history["pY"][epoch] = self.second_som.Y.reshape(self.n_second_node, self.n_first_node, self.dim)
                self.history["bmm"][epoch] = self.second_som.bmus


    def _fit_first_SOMs(self, epoch):
        for n, som in enumerate(self.first_soms):
            if epoch == 0:
                som._cooperative_process(epoch)
                som._adaptive_process()
                som._competitive_process()
            else:
                som.Y = self.second_som.Y[n].reshape(self.n_first_node, self.dim)  # copy back
                som._competitive_process()
                som._cooperative_process(epoch)
                som._adaptive_process()
            self.first_soms_mapping[n] = som.Y.reshape(self.n_first_node * self.dim)


    def _fit_second_SOM(self, epoch):
        self.second_som.X = self.first_soms_mapping
        self.second_som._cooperative_process(epoch)
        self.second_som._adaptive_process()
        self.second_som._competitive_process()
