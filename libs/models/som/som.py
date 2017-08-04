# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance as dist

class SOM:
    def __init__(self, X, latent_dim, resolution, sigma_max, sigma_min, tau, init='random'):
        self.X = X
        self.N = self.X.shape[0]

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.tau = tau

        if isinstance(init, str) and init == 'random':
            self.Z = np.random.rand(self.N, latent_dim) * 2.0 - 1.0
        elif isinstance(init, np.ndarray) and init.shape == (self.N, latent_dim):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        if latent_dim == 1:
            self.Zeta = np.linspace(-1.0, 1.0, resolution)[:,np.newaxis]
        elif latent_dim == 2:
            zeta = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
            self.Zeta = np.dstack(zeta).reshape(resolution**2, latent_dim)
        else:
            raise ValueError("invalid latent dimension: {}".format(latent_dim))

    def learning(self, t):
        # 協調過程
        # 学習量を計算
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * np.exp(-t / self.tau) # 近傍半径を設定
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