# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from libs.models.tsom import TSOM2
from libs.models.som import SOM

class TSOM_cross_SOM:
    def __init__(self, Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZinit, cZinit):

        self.Datasets = Datasets
        self.n_class = self.Datasets.shape[0]
        self.n_sample_1 = self.Datasets.shape[1]
        self.n_sample_2 = self.Datasets.shape[2]
        self.Dim = self.Datasets.shape[3]
        self.parent_latent_dim = parent_latent_dim
        # tsom 潜在空間の設定
        if isinstance(child_latent_dim, (list, tuple)):
            self.child_latent_dim_1 = child_latent_dim[0]
            self.child_latent_dim_2 = child_latent_dim[1]
        else:
            raise ValueError("invalid latent_dim: {}".format(child_latent_dim))
        self.parent_resolution = parent_resolution
        # tsom resolutionの設定
        if isinstance(child_resolution, (list, tuple)):
            self.child_resolution_1 = child_resolution[0]
            self.child_resolution_2 = child_resolution[1]
        else:
            raise ValueError("invalid resolution: {}".format(child_resolution))
        self.parent_sigma_max = parent_sigma_max
        self.child_sigma_max = child_sigma_max
        self.parent_sigma_min = parent_sigma_min
        self.child_sigma_min = child_sigma_min
        self.parent_tau = parent_tau
        self.child_tau = child_tau
        if pZinit is None:
            self.pZinit = 'random'
        else:
            self.pZinit = pZinit
        if cZinit is None:
            self.cZinit = 'random'
        else:
            self.cZinit = cZinit
        self.pK = parent_resolution ** parent_latent_dim
        # tsom の各モードのノード数
        self.cK1 = self.child_resolution_1 ** self.child_latent_dim_1
        self.cK2 = self.child_resolution_2 ** self.child_latent_dim_2
        # self.W = np.zeros((self.n_class, self.cK * self.Dim))
        self.children_to_parent = np.zeros((self.n_class, self.cK1 * self.cK2 * self.Dim))
        self.history = {}

        self._done_fit = False
        self.Z_grad = None

    def fit(self, nb_epoch, verbose=True):

        self.history['cZ1'] = np.zeros((nb_epoch, self.n_class, self.n_sample_1, self.child_latent_dim_1))
        self.history['cZ2'] = np.zeros((nb_epoch, self.n_class, self.n_sample_2, self.child_latent_dim_2))
        self.history['pZ'] = np.zeros((nb_epoch, self.n_class, self.parent_latent_dim))
        self.history['cY'] = np.zeros((nb_epoch, self.n_class, self.cK1, self.cK2, self.Dim))
        self.history['pY'] = np.zeros((nb_epoch, self.pK, self.cK1, self.cK2, self.Dim))
        self.history["bmu1"] = np.zeros((nb_epoch, self.n_class, self.n_sample_1))
        self.history["bmu2"] = np.zeros((nb_epoch, self.n_class, self.n_sample_2))
        self.history["bmm"] = np.zeros((nb_epoch, self.n_class))

        # ---------------- pairpro ---------------------- start
        # 子TSOM用意 データのclass 個数分
        tsom_children = []
        for data_number in range(self.n_class):
            tsom_children.append(
                TSOM2(self.Datasets[data_number], latent_dim=[self.child_latent_dim_1,self.child_latent_dim_2], resolution=[self.child_resolution_1,self.child_resolution_2], SIGMA_MAX=self.child_sigma_max, SIGMA_MIN=self.child_sigma_min, TAU=self.child_tau, init=self.cZinit)
            )

        # 親SOM用意
        # 親SOMに入力されるデータ：子SOMの数n_class＊子SOMのノード数K＊子SOMの入力データの次元D
        # とりあえず仮データ渡しておく
        provisional_data = np.zeros((self.n_class, self.cK1 * self.cK2 * self.Dim))
        som_parent = SOM(provisional_data, latent_dim=self.parent_latent_dim, resolution=self.parent_resolution, sigma_max=self.parent_sigma_max, sigma_min=self.parent_sigma_min, tau=self.parent_tau, init=self.pZinit)

        self.history['cZeta1'] = tsom_children[0].Zeta1
        self.history['cZeta2'] = tsom_children[0].Zeta2
        self.history['pZeta'] = som_parent.Zeta

        for epoch in tqdm(np.arange(nb_epoch)):
            # 子TSOMの学習
            # 初回：init で貰ったZ初期値からスタート epoch == 0
            # 2回目以降：親SOMからのコピーバックで貰った参照ベクトル初期値からスタート epoch > 0
            for children_number in range(self.n_class):
                if epoch == 0:
                    # 初回は協調過程かｋら
                    # def _cooperative_process(self, epoch):
                    tsom_children[children_number]._cooperative_process(epoch)
                    # def _adaptive_process(self):
                    tsom_children[children_number]._adaptive_process_nonmissing_indirect()
                    # def _competitive_process(self):
                    tsom_children[children_number]._competitive_process_nonmissing_indirect()
                else:
                    # コピーバックで親SOMの参照ベクトルを初期値としてもらう
                    # コピーバックベクトル: n_class*(K*D)
                    tsom_children[children_number].Y = som_parent.Y[children_number].reshape(self.cK1, self.cK2, self.Dim)
                    # ２回目以降は競合過程から
                    tsom_children[children_number]._competitive_process_nonmissing_indirect()
                    tsom_children[children_number]._cooperative_process(epoch)
                    tsom_children[children_number]._adaptive_process_nonmissing_indirect()

            # 親SOMのデータ入力
            # 子TSOMの参照ベクトルを１次元に成型してくっつけて渡す
            for children_number in range(self.n_class):
                self.children_to_parent[children_number] = tsom_children[children_number].Y.reshape(self.cK1 * self.cK2 * self.Dim)
            som_parent.X = self.children_to_parent

            # 親SOMの学習
            som_parent._cooperative_process(epoch)
            som_parent._adaptive_process()
            som_parent._competitive_process()

        # ---------------- pairpro ---------------------- end

            for n in range(self.n_class):
                self.history["cZ1"][epoch, n] = tsom_children[n].Z1
                self.history["cZ2"][epoch, n] = tsom_children[n].Z2
                self.history["cY"][epoch, n] = tsom_children[n].Y
                self.history["bmu1"][epoch, n] = tsom_children[n].k_star1
                self.history["bmu2"][epoch, n] = tsom_children[n].k_star2
            self.history["pZ"][epoch] = som_parent.Z
            self.history["pY"][epoch] = som_parent.Y.reshape(self.pK, self.cK1, self.cK2, self.Dim)
            self.history["bmm"][epoch] = som_parent.bmus
            # for n in range(self.n_class):
            #     self.history["cZ"][epoch, n] = SOMs[n].Z
            #     self.history["cY"][epoch, n] = SOMs[n].Y
            #     self.history["bmu"][epoch, n] = SOMs[n].bmus
            # self.history["pZ"][epoch] = SOM.Z
            # self.history["pY"][epoch] = SOM.Y.reshape(self.pK, self.cK, self.Dim)
            # self.history["bmm"][epoch] = SOM.bmus
