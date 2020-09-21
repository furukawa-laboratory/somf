# -*- coding: utf-8 -*-
import numpy as np

from tqdm import tqdm


class SOMUseFor:
    def __init__(self, X, latent_dim, resolution, sigma_max, sigma_min, tau, init='random'):
        self.X = X #観測データ

        # 近傍半径のスケジュールに関するパラメータ
        self.sigma_max = sigma_max # 最大近傍半径
        self.sigma_min = sigma_min # 最小近傍半径
        self.tau = tau # 時定数

        # その他のパラメータ
        self.N = self.X.shape[0] # 観測データ数
        self.K = resolution * resolution # ノード数
        self.D = X.shape[1] # 観測空間の次元
        self.L = latent_dim # 潜在空間の次元

        # 潜在空間の定義
        if latent_dim == 1:
            self.Zeta = np.linspace(-1.0, 1.0, resolution)[:,np.newaxis]
            # [-1.0,+1.0]で等間隔に配置
        elif latent_dim == 2:
            zeta = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
            self.Zeta = np.dstack(zeta).reshape(resolution**2, latent_dim)
            # 各次元[-1.0,1.0]で等間隔に配置し，それらの組み合わせを全て網羅
        else:
            raise ValueError("invalid latent dimension: {}".format(latent_dim))
            # 例外処理

        self.K = resolution**self.L

        # 勝者ノードの初期化
        if isinstance(init, str) and init == 'random':
            self.Z = np.random.rand(self.N, latent_dim) * 2.0 - 1.0
            # init = 'random'が指定されたら[-1.0,1.0]の範囲で乱数で決定
        elif isinstance(init, str) and init == 'random_bmu':
            init_bmus = np.random.randint(0, Zeta.shape[0] - 1, self.N)
            self.Z = self.Zeta[init_bmus,:]
        elif isinstance(init, np.ndarray) and init.shape == (self.N, latent_dim):
            self.Z = init.copy()
            # サイズに矛盾がない配列が与えられればそれを勝者ノードの初期値とする
        else:
            raise ValueError("invalid init: {}".format(init))
            # 例外処理

        self.history = {}


    def fit(self, nb_epoch=100, verbose=True):

        self.history['z'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y'] = np.zeros((nb_epoch, self.K, self.D))

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)
        for epoch in bar:
        # 協調過程
        # 学習量の計算
            H = np.zeros((self.K,self.N)) # 全ての学習量を保持する配列を用意
            # 近傍半径の計算
            sigma = max(self.sigma_min, self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - (epoch / self.tau)))# 線形で減少
            #sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * ( 1 - (epoch / self.tau) ) # 指数的に減少
            for k in range(self.K):
                for n in range(self.N):
                    dist = 0 # 距離を初期化
                    for l in range(self.L):
                        dist += (self.Z[n,l] - self.Zeta[k,l]) ** 2.0
                        # k番目のノードとn番目の勝者ノードの距離を求める
                    H[k,n] = np.exp(-dist / (2*sigma*sigma))
                    # k番目のノードがn番目の勝者ノードから受ける学習量を計算

            # 適応過程
            # 参照ベクトルの更新
            self.Y = np.zeros((self.K,self.D)) # 参照ベクトルを保持する配列を作成
            for k in range(self.K):
                g = 0
                for n in range(self.N):
                    g += H[k,n] # k番目のノードが持つ学習量の総和を計算
                for d in range(self.D):
                    for n in range(self.N):
                        self.Y[k,d] += H[k,n] * self.X[n,d] # 学習量で重み付けしたxの総和を計算
                    self.Y[k,d] /= g # gで割って規格化

            # 競合過程
            # 各観測データに対する勝者ノードを求める
            for n in range(self.N):
                bmu = 0 #勝者ノード番号を初期化
                dist_min = np.float_('inf') # 現時点での最小距離を初期化
                for k in range(self.K):
                    dist = 0
                    for d in range(self.D):
                        dist += (self.X[n,d] - self.Y[k,d]) ** 2.0
                        # n番目の観測データとk番目の参照ベクトルの距離を計算
                    if dist < dist_min:
                        # 現時点での最小距離よりも求めた距離が小さければ
                        bmu = k # 勝者ノード番号を更新
                        dist_min = dist # 現時点での最小距離を初期化
                self.Z[n,:] = self.Zeta[bmu, :] # 最終的な勝者ノード番号を用いて勝者ノードの座標を求める

            self.history['z'][epoch] = self.Z
            self.history['y'][epoch] = self.Y
