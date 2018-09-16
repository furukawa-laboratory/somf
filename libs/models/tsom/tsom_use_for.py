import numpy as np
from tqdm import tqdm
from libs.tools.create_zeta import create_zeta
import random


class TSOM2():
    def __init__(self, X, latent_dim, resolution, SIGMA_MAX, SIGMA_MIN, TAU, init='random'):

        # 入力データXについて
        if X.ndim == 2:
            self.X = X.reshape((X.shape[0], X.shape[1], 1))
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.observed_dim = self.X.shape[2]  # 観測空間の次元

        elif X.ndim == 3:
            self.X = X
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.observed_dim = self.X.shape[2]  # 観測空間の次元
        else:
            raise ValueError("invalid X: {}\nX must be 2d or 3d ndarray".format(X))

        # 最大近傍半径(SIGMAX)の設定
        if type(SIGMA_MAX) is float:
            self.SIGMA1_MAX = SIGMA_MAX
            self.SIGMA2_MAX = SIGMA_MAX
        elif isinstance(SIGMA_MAX, (list, tuple)):
            self.SIGMA1_MAX = SIGMA_MAX[0]
            self.SIGMA2_MAX = SIGMA_MAX[1]
        else:
            raise ValueError("invalid SIGMA_MAX: {}".format(SIGMA_MAX))

        # 最小近傍半径(SIGMA_MIN)の設定
        if type(SIGMA_MIN) is float:
            self.SIGMA1_MIN = SIGMA_MIN
            self.SIGMA2_MIN = SIGMA_MIN
        elif isinstance(SIGMA_MIN, (list, tuple)):
            self.SIGMA1_MIN = SIGMA_MIN[0]
            self.SIGMA2_MIN = SIGMA_MIN[1]
        else:
            raise ValueError("invalid SIGMA_MIN: {}".format(SIGMA_MIN))

        # 時定数(TAU)の設定
        if type(TAU) is int:
            self.TAU1 = TAU
            self.TAU1 = TAU
        elif isinstance(TAU, (list, tuple)):
            self.TAU1 = TAU[0]
            self.TAU2 = TAU[1]
        else:
            raise ValueError("invalid TAU: {}".format(TAU))

        # resolutionの設定
        if type(resolution) is int:
            resolution1 = resolution
            resolution2 = resolution
        elif isinstance(resolution, (list, tuple)):
            resolution1 = resolution[0]
            resolution2 = resolution[1]
        else:
            raise ValueError("invalid resolution: {}".format(resolution))

        # 潜在空間の設定
        if type(latent_dim) is int:  # latent_dimがintであればどちらのモードも潜在空間の次元は同じ
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim

        elif isinstance(latent_dim, (list, tuple)):
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))
            # latent_dimがlist,float,3次元以上はエラーかな?
        self.Zeta1 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim1, resolution=resolution1, include_min_max=True)
        self.Zeta2 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim2, resolution=resolution2, include_min_max=True)

        # K1とK2は潜在空間の設定が終わった後がいいよね
        self.K1 = self.Zeta1.shape[0]
        self.K2 = self.Zeta2.shape[0]

        # 勝者ノードの初期化
        self.Z1 = None
        self.Z2 = None
        if isinstance(init, str) and init in 'random':
            self.Z1 = np.random.rand(self.N1, self.latent_dim1) * 2.0 - 1.0
            self.Z2 = np.random.rand(self.N2, self.latent_dim2) * 2.0 - 1.0
        elif isinstance(init, (tuple, list)) and len(init) == 2:
            if isinstance(init[0], np.ndarray) and init[0].shape == (self.N1, self.latent_dim1):
                self.Z1 = init[0].copy()
            else:
                raise ValueError("invalid inits[0]: {}".format(init))
            if isinstance(init[1], np.ndarray) and init[1].shape == (self.N2, self.latent_dim2):
                self.Z2 = init[1].copy()
            else:
                raise ValueError("invalid inits[1]: {}".format(init))
        else:
            raise ValueError("invalid inits: {}".format(init))

        #モデルの初期化
        self.U = np.zeros((self.N1, self.K2, self.observed_dim))
        self.V = np.zeros((self.K1, self.N2, self.observed_dim))
        self.Y = np.zeros((self.K1, self.K2, self.observed_dim))



        self.history = {}

    def fit(self, nb_epoch=200):
        self.history['y'] = np.zeros((nb_epoch, self.K1, self.K2, self.observed_dim))
        self.history['z1'] = np.zeros((nb_epoch, self.N1, self.latent_dim1))
        self.history['z2'] = np.zeros((nb_epoch, self.N2, self.latent_dim2))
        self.history['sigma1'] = np.zeros(nb_epoch)
        self.history['sigma2'] = np.zeros(nb_epoch)

        #勝者番号の初期化
        k_star = np.arange(self.K1)
        l_star = np.arange(self.K2)

        for epoch in tqdm(np.arange(nb_epoch)):
            # 協調過程
            h1 = np.zeros((self.N1, self.K1))
            h2 = np.zeros((self.N2, self.K2))

            # mode1の学習量の計算
            sigma1 = self.SIGMA1_MIN + (self.SIGMA1_MAX - self.SIGMA1_MIN) * np.exp(-epoch / self.TAU1)
            for i in np.arange(self.N1):
                for k in np.arange(self.K1):
                    zeta_dis1 = 0
                    for latent_l in np.arange(self.latent_dim1):
                        zeta_dis1 += (self.Z1[k_star[i]][latent_l] - self.Zeta1[k][latent_l]) ** 2
                    h1[i][k] = np.exp(-0.5 * (zeta_dis1 * zeta_dis1) / sigma1 ** 2)

            # mode2の学習量の計算
            sigma2 = self.SIGMA2_MIN + (self.SIGMA2_MAX - self.SIGMA2_MIN) * np.exp(-epoch / self.TAU2)
            for j in np.arange(self.N2):
                for l in np.arange(self.K2):
                    zeta_dis2 = 0
                    for latent_l in np.arange(self.latent_dim2):
                        zeta_dis2 += (self.Z2[l_star[j]][latent_l] - self.Zeta2[l][latent_l]) ** 2
                    h2[j][l] = np.exp(-0.5 * (zeta_dis2 * zeta_dis2) / sigma2 ** 2)

            # 適応過程の計算
            # gの計算
            # mode1のgの計算
            for k in np.arange(self.K1):
                g1 = 0
                for i in np.arange(self.N1):
                    g1 += h1[i][k]
                for i in np.arange(self.N1):
                    h1[i][k] /= g1

            # mode2のgの計算
            for l in np.arange(self.K2):
                g2 = 0
                for j in np.arange(self.N2):
                    g2 += h2[j][l]
                for j in np.arange(self.N2):
                    h2[j][l] /= g2

            # モデルの更新
            # 1次モデル
            self.U = np.zeros((self.N1, self.K2, self.observed_dim))
            self.V = np.zeros((self.K1, self.N2, self.observed_dim))
            self.Y = np.zeros((self.K1, self.K2, self.observed_dim))
            for i in np.arange(self.N1):
                for l in np.arange(self.K2):
                    for d in np.arange(self.observed_dim):
                        for j in np.arange(self.N2):
                            self.U[i][l][d] += h2[j][l] * self.X[i][j][d]

            for k in np.arange(self.K1):
                for j in np.arange(self.N2):
                    for d in np.arange(self.observed_dim):
                        for i in np.arange(self.N1):
                            self.V[k][j][d] += h1[i][k] * self.X[i][j][d]

            # 2次モデルの更新
            for k in np.arange(self.K1):
                for l in np.arange(self.K2):
                    for d in np.arange(self.observed_dim):
                        for i in np.arange(self.N1):
                            for j in np.arange(self.N2):
                                self.Y[k][l][d] += h1[i][k] * h2[j][l] * self.X[i][j][d]

            #距離行列の初期化
            mode1_D = np.zeros((self.N1, self.K1))
            mode2_D = np.zeros((self.N2, self.K2))

            # 競合過程を作る
            # mode1の競合過程
            for i in np.arange(self.N1):
                for k in np.arange(self.K1):
                    distance2 = 0
                    for l in np.arange(self.K2):
                        distance = 0
                        for d in np.arange(self.observed_dim):
                            distance += (self.U[i][l][d] - self.Y[k][l][d]) ** 2
                        distance2 += distance
                    mode1_D[i][k] = distance2

            k_star = np.argmin(mode1_D, axis=1)

            # mode2の競合過程
            for j in np.arange(self.N2):
                for l in np.arange(self.K2):
                    distance2 = 0
                    for k in np.arange(self.K1):
                        distance = 0
                        for d in np.arange(self.observed_dim):
                            distance += (self.V[k][j][d] - self.Y[k][l][d]) ** 2
                        distance2 += distance
                    mode2_D[j][l] = distance2

            l_star = np.argmin(mode2_D, axis=1)


            self.history['y'][epoch, :, :] = self.Y
            self.history['z1'][epoch, :] = self.Z1
            self.history['z2'][epoch, :] = self.Z2
            self.history['sigma1'][epoch] = sigma1
            self.history['sigma2'][epoch] = sigma2




        # for epoch in tqdm(np.arange(nb_epoch)):
        #     # 競合過程を作る
        #     # mode1の競合過程
        #     for i in np.arange(self.N1):
        #         for k in np.arange(self.K1):
        #             distance2 = 0
        #             for l in np.arange(self.K2):
        #                 distance = 0
        #                 for d in np.arange(self.observed_dim):
        #                     distance += (U[i][l][d] - Y[k][l][d]) ** 2
        #                 distance2 += distance
        #             mode1_D[i][k] = distance2
        #
        #     k_star = np.argmin(mode1_D, axis=1)
        #
        #     # mode2の競合過程
        #     for j in np.arange(self.N2):
        #         for l in np.arange(self.K2):
        #             distance2 = 0
        #             for k in np.arange(self.K1):
        #                 distance = 0
        #                 for d in np.arange(self.observed_dim):
        #                     distance += (V[k][j][d] - Y[k][l][d]) ** 2
        #                 distance2 += distance
        #             mode2_D[j][l] = distance2
        #
        #     l_star = np.argmin(mode2_D, axis=1)
        #
        #     # 協調過程
        #
        #     h1 = np.zeros((self.N1, self.K1))
        #     h2 = np.zeros((self.N2, self.K2))
        #
        #     # mode1の学習量の計算
        #     sigma1 = self.SIGMA1_MIN + (self.SIGMA1_MAX - self.SIGMA1_MIN) * np.exp(-epoch / self.TAU1)
        #     for i in np.arange(self.N1):
        #         for k in np.arange(self.K1):
        #             zeta_dis1 = 0
        #             for latent_l in np.arange(self.latent_dim1):
        #                 zeta_dis1 += (Zeta1[k_star[i]][latent_l] - Zeta1[k][latent_l]) ** 2
        #             h1[i][k] = np.exp(-0.5 * (zeta_dis1 * zeta_dis1) / sigma1 ** 2)
        #
        #     # mode2の学習量の計算
        #     sigma2 = self.SIGMA2_MIN + (self.SIGMA2_MAX - self.SIGMA2_MIN) * np.exp(-epoch / self.TAU2)
        #     for j in np.arange(self.N2):
        #         for l in np.arange(self.K2):
        #             zeta_dis2 = 0
        #             for latent_l in np.arange(self.latent_dim2):
        #                 zeta_dis2 += (Zeta2[l_star[j]][latent_l] - Zeta2[l][latent_l]) ** 2
        #             h2[j][l] = np.exp(-0.5 * (zeta_dis2 * zeta_dis2) / sigma2 ** 2)
        #
        #     # 適応過程の計算
        #     # gの計算
        #     # mode1のgの計算
        #     for k in np.arange(self.K1):
        #         g1 = 0
        #         for i in np.arange(self.N1):
        #             g1 += h1[i][k]
        #         for i in np.arange(self.N1):
        #             h1[i][k] /= g1
        #
        #     # mode2のgの計算
        #     for l in np.arange(self.K2):
        #         g2 = 0
        #         for j in np.arange(self.N2):
        #             g2 += h2[j][l]
        #         for j in np.arange(self.N2):
        #             h2[j][l] /= g2
        #
        #     # モデルの更新
        #     # 1次モデル
        #     U = np.zeros((self.N1, self.K2, self.observed_dim))
        #     V = np.zeros((self.K1, self.N2, self.observed_dim))
        #     Y = np.zeros((self.K1, self.K2, self.observed_dim))
        #     for i in np.arange(self.N1):
        #         for l in np.arange(self.K2):
        #             for d in np.arange(self.observed_dim):
        #                 for j in np.arange(self.N2):
        #                     U[i][l][d] += h2[j][l] * X[i][j][d]
        #
        #     for k in np.arange(self.K1):
        #         for j in np.arange(self.N2):
        #             for d in np.arange(self.observed_dim):
        #                 for i in np.arange(self.N1):
        #                     V[k][j][d] += h1[i][k] * X[i][j][d]
        #
        #     # 2次モデルの更新
        #     for k in np.arange(self.K1):
        #         for l in np.arange(self.K2):
        #             for d in np.arange(self.observed_dim):
        #                 for i in np.arange(self.N1):
        #                     for j in np.arange(self.N2):
        #                         Y[k][l][d] += h1[i][k] * h2[j][l] * X[i][j][d]
        #     Y_allepoch[epoch, :, :] = Y
        #     self.history['y'][epoch, :, :] = self.Y
        #     self.history['z1'][epoch, :] = self.Z1
        #     self.history['z2'][epoch, :] = self.Z2
        #     self.history['sigma1'][epoch] = sigma1
        #     self.history['sigma2'][epoch] = sigma2
