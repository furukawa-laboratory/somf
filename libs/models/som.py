# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance as dist
from tqdm import tqdm
from sklearn.decomposition import PCA


class SOM:
    def __init__(self, X, latent_dim, resolution, sigma_max, sigma_min, tau, init='random'):
        self.X = X
        self.N = self.X.shape[0]

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau

        self.D = X.shape[1]
        self.L = latent_dim

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
            self.Z = pca.fit_transform(X)
        elif isinstance(init, np.ndarray) and init.shape == (self.N, latent_dim):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        self.history = {}

    def fit(self, nb_epoch=100, verbose=True):

        self.history['z'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y'] = np.zeros((nb_epoch, self.K, self.D))
        self.history['sigma'] = np.zeros(nb_epoch)

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        for epoch in bar:
            # 協調過程
            # 学習量を計算
            # sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * np.exp(-epoch / self.tau) # 近傍半径を設定
            sigma = max(self.sigma_min, self.sigma_max * ( 1 - (epoch / self.tau) ) )# 近傍半径を設定
            Dist = dist.cdist(self.Zeta, self.Z, 'sqeuclidean')
            # KxNの距離行列を計算
            # ノードと勝者ノードの全ての組み合わせにおける距離を網羅した行列
            H = np.exp(-Dist / (2 * sigma * sigma)) # KxNの学習量行列を計算

            # 適合過程
            # 参照ベクトルの更新
            G = np.sum(H, axis=1)[:, np.newaxis] # 各ノードが受ける学習量の総和を保持するKx1の列ベクトルを計算
            Ginv = np.reciprocal(G) # Gのそれぞれの要素の逆数を取る
            R = H * Ginv # 学習量の総和が1になるように規格化
            self.Y = R @ self.X # 学習量を重みとして観測データの平均を取り参照ベクトルとする

            # 競合過程
            # 勝者ノードの計算
            Dist = dist.cdist(self.X, self.Y) #NxKの距離行列を計算
            bmus = Dist.argmin(axis=1)
            # Nx1の勝者ノード番号をまとめた列ベクトルを計算
            # argmin(axis=1)を用いて各行で最小値を探しそのインデックスを返す
            self.Z = self.Zeta[bmus, :] # 勝者ノード番号から勝者ノードを求める

            self.history['z'][epoch] = self.Z
            self.history['y'][epoch] = self.Y
            self.history['sigma'][epoch] = sigma
