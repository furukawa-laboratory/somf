# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from libs.models.som import SOM as som

class SOM2_harada:
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
        # self.W = np.zeros((self.n_class, self.cK * self.Dim))
        self.children_to_parent = np.zeros((self.n_class, self.cK * self.Dim))
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

        # ---------------- pairpro ---------------------- start
        # 子SOM用意 データのclass 個数分
        som_children = []
        for data_number in range(self.n_class):
            som_children.append(
                som(self.Datasets[data_number], latent_dim=self.child_latent_dim, resolution=self.child_resolution, sigma_max=self.child_sigma_max, sigma_min=self.child_sigma_min, tau=self.child_tau, init=self.cZinit)
            )

        # 親SOM用意
        # 親SOMに入力されるデータ：子SOMの数n_class＊子SOMのノード数K＊子SOMの入力データの次元D
        # とりあえず仮データ渡しておく
        provisional_data = np.zeros((self.n_class, self.cK * self.Dim))
        som_parent = som(provisional_data, latent_dim=self.parent_latent_dim, resolution=self.parent_resolution, sigma_max=self.parent_sigma_max, sigma_min=self.parent_sigma_min, tau=self.parent_tau, init=self.pZinit)

        self.history['cZeta'] = som_children[0].Zeta
        self.history['pZeta'] = som_parent.Zeta

        for epoch in tqdm(np.arange(nb_epoch)):
            # 子SOMの学習
            # 初回：init で貰ったZ初期値からスタート epoch == 0
            # 2回目以降：親SOMからのコピーバックで貰った参照ベクトル初期値からスタート epoch > 0
            for children_number in range(self.n_class):
                if epoch == 0:
                    # 初回は協調過程かｋら
                    # def _cooperative_process(self, epoch):
                    som_children[children_number]._cooperative_process(epoch)
                    # def _adaptive_process(self):
                    som_children[children_number]._adaptive_process()
                    # def _competitive_process(self):
                    som_children[children_number]._competitive_process()
                else:
                    # コピーバックで親SOMの参照ベクトルを初期値としてもらう
                    # コピーバックベクトル: n_class*(K*D)
                    som_children[children_number].Y = som_parent.Y[children_number].reshape(self.cK, self.Dim)
                    # ２回目以降は競合過程から
                    som_children[children_number]._competitive_process()
                    som_children[children_number]._cooperative_process(epoch)
                    som_children[children_number]._adaptive_process()

            # 親SOMのデータ入力
            # 子SOMの参照ベクトルを１次元に成型してくっつけて渡す
            for children_number in range(self.n_class):
                self.children_to_parent[children_number] = som_children[children_number].Y.reshape(self.cK * self.Dim)
            som_parent.X = self.children_to_parent

            # 親SOMの学習
            som_parent._cooperative_process(epoch)
            som_parent._adaptive_process()
            som_parent._competitive_process()

        # ---------------- pairpro ---------------------- end

            for n in range(self.n_class):
                self.history["cZ"][epoch, n] = som_children[n].Z
                self.history["cY"][epoch, n] = som_children[n].Y
                self.history["bmu"][epoch, n] = som_children[n].bmus
            self.history["pZ"][epoch] = som_parent.Z
            self.history["pY"][epoch] = som_parent.Y.reshape(self.pK, self.cK, self.Dim)
            self.history["bmm"][epoch] = som_parent.bmus
            # for n in range(self.n_class):
            #     self.history["cZ"][epoch, n] = SOMs[n].Z
            #     self.history["cY"][epoch, n] = SOMs[n].Y
            #     self.history["bmu"][epoch, n] = SOMs[n].bmus
            # self.history["pZ"][epoch] = SOM.Z
            # self.history["pY"][epoch] = SOM.Y.reshape(self.pK, self.cK, self.Dim)
            # self.history["bmm"][epoch] = SOM.bmus
