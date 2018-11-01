import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib.artist as artist
import matplotlib.image as image
from sklearn.preprocessing import StandardScaler

import scipy.spatial.distance as dist

np.random.seed(2)

class SqTSOM2_Umatrix:
    def __init__(self, z1=None, z2=None, x=None, sigma1=0.2, sigma2=0.2, resolution=100, labels=None, fig_size=[15, 15], cmap_type='jet'):
        # set input
        self.X = x
        I = self.X.shape[0]
        J = self.X.shape[1]
        self.X=self.X.reshape((I,J))
        self.X2 = self.X.T
        self.Z1 = z1
        self.Z2 = z2
        self.Sigma1 = sigma1
        self.Sigma2 = sigma2
        self.Resolution = resolution
        self.K = resolution * resolution
        self.N = self.X.shape[0]
        self.N2 = self.X.shape[1]
        self.L = self.Z1.shape[1]

        # 描画キャンバスの設定
        self.Fig1 = plt.figure(figsize=(fig_size[0], fig_size[1]))
        self.Map1 = self.Fig1.add_subplot(1, 1, 1)
        self.Cmap_type = cmap_type
        self.labels = labels

        # 潜在空間の代表点の設定
        self.Zeta1 = np.meshgrid(np.linspace(self.Z1[:, 0].min(), self.Z1[:, 0].max(), self.Resolution),
                           np.linspace(self.Z1[:, 1].min(), self.Z1[:, 1].max(), self.Resolution))
        self.Zeta1 = np.dstack(self.Zeta1).reshape(self.K, self.L)

        self.Zeta2 = np.meshgrid(np.linspace(self.Z2[:, 0].min(), self.Z2[:, 0].max(), self.Resolution),
                                np.linspace(self.Z2[:, 1].min(), self.Z2[:, 1].max(), self.Resolution))
        self.Zeta2 = np.dstack(self.Zeta2).reshape(self.K, self.L)


    # U-matrix表示
    def draw_umatrix(self):
        # U-matrix表示用の値を算出
        dY_std = self.__calc_umatrix()
        dY2_std = self.__calc_umatrix2()
        U_matrix_val = dY_std.reshape((self.Resolution, self.Resolution))
        U_matrix_val2 = dY2_std.reshape((self.Resolution, self.Resolution))
        # Square Tensor SOM
        U_matrix_val = (U_matrix_val + U_matrix_val2)/2

        # U-matrix表示
        self.Map1.set_title("U-matrix1")
        if self.Zeta1[:, 0].min() != self.Zeta1[:, 0].max():
            self.Map1.imshow(U_matrix_val, interpolation='spline36',
                             extent=[self.Zeta1[:, 0].min(), self.Zeta1[:, 0].max(), self.Zeta1[:, 1].max(),
                                     self.Zeta1[:, 1].min()], cmap=self.Cmap_type, vmax=0.5, vmin=-0.5)
        else:
            self.Map1.imshow(U_matrix_val.T, interpolation='spline36',
                             extent=[self.Zeta1[:, 1].min(),
                                     self.Zeta1[:, 1].max(), -1.3, -0.7], cmap=self.Cmap_type, vmax=0.5, vmin=-0.5)

        if self.Zeta1[:, 0].min() != self.Zeta1[:, 0].max():
            self.Map1.scatter(x=self.Z1[:, 0], y=self.Z1[:, 1], c='k')
        else:
            self.Map1.scatter(x=self.Z1[:, 1], y=self.Z1[:, 0], c='k')

        umatrix_val = (np.random.rand(20, 20) - 0.5) * 2
        if self.labels is None:
            self.labels = np.arange(self.N)

        # 勝者位置が重なった時用の処理
        if self.Zeta1[:, 0].min() != self.Zeta1[:, 0].max():
            epsilon = 0.02 * (self.Z1.max() - self.Z1.min())
            for i in range(self.Z1.shape[0]):
                count = 0
                for j in range(i):
                    if np.allclose(self.Z1[j, :], self.Z1[i, :]):
                        count += 1
                self.Map1.text(self.Z1[i, 0] - 0.01, self.Z1[i, 1] + epsilon * count, self.labels[i], fontsize=12,
                               color='k')
        else:
            # epsilon = 0.02 * (self.Z1.max() - self.Z1.min())
            epsilon = 0.03
            for i in range(self.Z1.shape[0]):
                count = 0
                for j in range(i):
                    if np.allclose(self.Z1[j, :], self.Z1[i, :]):
                        count += 1
                if i // 2 != 0:
                    self.Map1.text(self.Z1[i, 1] + 0.01, self.Z1[i, 0] + epsilon * count, self.labels[i], fontsize=12,
                                   color='k')
                else:
                    self.Map1.text(self.Z1[i, 1] + 0.01, self.Z1[i, 0] - epsilon * count, self.labels[i], fontsize=12,
                                   color='k')
        print('finish')
        plt.show()


    # U-matrix表示用の値（勾配）を算出
    def __calc_umatrix(self):
        # H, G, Rの算出
        dist_z = dist.cdist(self.Zeta1, self.Z1, 'sqeuclidean')
        H = np.exp(-dist_z / (2 * self.Sigma1 * self.Sigma1))
        G = H.sum(axis=1)[:, np.newaxis]
        R = H / G

        # V, V_meanの算出
        V = R[:, :, np.newaxis] * (self.Z1[np.newaxis, :, :] - self.Zeta1[:, np.newaxis, :])          # KxNxL
        V_mean = V.sum(axis=1)[:, np.newaxis, :]                                                    # Kx1xL

        # dYdZの算出
        dRdZ = V - R[:, :, np.newaxis] * V_mean                                                     # KxNxL
        dYdZ = np.sum(dRdZ[:, :, :, np.newaxis] * self.X[np.newaxis, :, np.newaxis, :], axis=1)     # KxLxD X:N*N2*D
        dYdZ_norm = np.sum(dYdZ ** 2, axis=(1, 2))                                                  # Kx1

        # 表示用の値を算出（標準化）
        sc = StandardScaler()
        dY_std = sc.fit_transform(dYdZ_norm[:, np.newaxis])
        umat_val = dY_std / 4.0

        umat_val[umat_val >= 0.5] = 0.5
        umat_val[umat_val <= -0.5] = -0.5

        return umat_val

    def __calc_umatrix2(self):
        # H, G, Rの算出
        dist_z = dist.cdist(self.Zeta2, self.Z2, 'sqeuclidean')
        H = np.exp(-dist_z / (2 * self.Sigma2 * self.Sigma2))
        G = H.sum(axis=1)[:, np.newaxis]
        R = H / G

        # V, V_meanの算出
        V = R[:, :, np.newaxis] * (self.Z2[np.newaxis, :, :] - self.Zeta2[:, np.newaxis, :])  # KxNxL
        V_mean = V.sum(axis=1)[:, np.newaxis, :]  # Kx1xL

        # dYdZの算出
        dRdZ = V - R[:, :, np.newaxis] * V_mean  # KxNxL
        dYdZ = np.sum(dRdZ[:, :, :, np.newaxis] * self.X2[np.newaxis, :, np.newaxis, :], axis=1)  # KxLxD
        dYdZ_norm = np.sum(dYdZ ** 2, axis=(1, 2))  # Kx1

        # 表示用の値を算出（標準化）
        sc = StandardScaler()
        dY_std = sc.fit_transform(dYdZ_norm[:, np.newaxis])
        umat_val = dY_std / 4.0

        umat_val[umat_val >= 0.5] = 0.5
        umat_val[umat_val <= -0.5] = -0.5

        return umat_val