# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance as dist
from tqdm import tqdm
from sklearn.decomposition import PCA


class SOM:
    def __init__(self, X, latent_dim, resolution, sigma_max, sigma_min, tau, init='random', metric="sqeuclidean"):
        self.X = X
        self.N = self.X.shape[0]

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau

        self.D = X.shape[1]
        self.L = latent_dim

        self.history = {}

        if isinstance(init, str) and init == 'PCA':
            pca = PCA(n_components=latent_dim)
            pca.fit(X)

        if latent_dim == 1:
            self.Zeta = np.linspace(-1.0, 1.0, resolution)[:, np.newaxis]
        elif latent_dim == 2:
            if isinstance(init, str) and init == 'PCA':
                comp1, comp2 = pca.singular_values_[0], pca.singular_values_[1]
                zeta = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-comp2/comp1, comp2/comp1, resolution))
            else:
                zeta = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
            self.Zeta = np.dstack(zeta).reshape(resolution**2, latent_dim)
        else:
            raise ValueError("invalid latent dimension: {}".format(latent_dim))

        self.K = resolution**self.L

        if isinstance(init, str) and init == 'random':
            self.Z = np.random.rand(self.N, latent_dim) * 2.0 - 1.0
        elif isinstance(init, str) and init == 'random_bmu':
            init_bmus = np.random.randint(0, self.Zeta.shape[0] - 1, self.N)
            self.Z = self.Zeta[init_bmus,:]
        elif isinstance(init, str) and init == 'PCA':
            self.Z = pca.transform(X)/comp1
        elif isinstance(init, np.ndarray) and init.dtype == int:
            init_bmus = init.copy()
            self.Z = self.Zeta[init_bmus, :]
        elif isinstance(init, np.ndarray) and init.shape == (self.N, latent_dim):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        #metricに関する処理
        if metric == "sqeuclidean":
            self.metric="sqeuclidean"

        elif metric == "KLdivergence":
            self.metric = "KLdivergence"
        else:
            raise ValueError("invalid metric: {}".format(metric))

        self.history = {}

    def _cooperative_process(self, epoch):
        # 学習量を計算
        # sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * np.exp(-epoch / self.tau) # 近傍半径を設定
        self.sigma = max(self.sigma_min, self.sigma_max * (1 - (epoch / self.tau)))  # 近傍半径を設定
        Dist = dist.cdist(self.Zeta, self.Z, 'sqeuclidean')
        # KxNの距離行列を計算
        # ノードと勝者ノードの全ての組み合わせにおける距離を網羅した行列
        self.H = np.exp(-Dist / (2 * self.sigma * self.sigma))  # KxNの学習量行列を計算

    def _adaptive_process(self):
        # 参照ベクトルの更新
        G = np.sum(self.H, axis=1)[:, np.newaxis]  # 各ノードが受ける学習量の総和を保持するKx1の列ベクトルを計算
        Ginv = np.reciprocal(G)  # Gのそれぞれの要素の逆数を取る
        R = self.H * Ginv  # 学習量の総和が1になるように規格化
        self.Y = R @ self.X  # 学習量を重みとして観測データの平均を取り参照ベクトルとする

    def _competitive_process(self):
        if self.metric is "sqeuclidean":  # ユークリッド距離を使った勝者決定
            # 勝者ノードの計算H
            Dist = dist.cdist(self.X, self.Y)  # NxKの距離行列を計算
            self.bmus = Dist.argmin(axis=1)
            # Nx1の勝者ノード番号をまとめた列ベクトルを計算
            # argmin(axis=1)を用いて各行で最小値を探しそのインデックスを返す
            self.Z = self.Zeta[self.bmus, :]  # 勝者ノード番号から勝者ノードを求める
        elif self.metric is "KLdivergence":  # KL情報量を使った勝者決定
            Dist = np.sum(self.X[:, np.newaxis, :] * np.log(self.Y)[np.newaxis, :, :], axis=2)  # N*K行列
            # 勝者番号の決定
            self.bmus = np.argmax(Dist, axis=1)
            # Nx1の勝者ノード番号をまとめた列ベクトルを計算
            # argmin(axis=1)を用いて各行で最小値を探しそのインデックスを返す
            self.Z = self.Zeta[self.bmus, :]  # 勝者ノード番号から勝者ノードを求める

    def update_mapping(self, Y):
        self.Y = Y

    def fit(self, nb_epoch=100, verbose=True):

        self.history['z'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y'] = np.zeros((nb_epoch, self.K, self.D))
        self.history['sigma'] = np.zeros(nb_epoch)

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        for epoch in bar:
            self._cooperative_process(epoch)   # 協調過程
            self._adaptive_process()           # 適合過程
            self._competitive_process()        # 競合過程

            self.history['z'][epoch] = self.Z
            self.history['y'][epoch] = self.Y
            self.history['sigma'][epoch] = self.sigma
