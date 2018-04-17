import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from sklearn.preprocessing import StandardScaler

class SOM_Umatrix:
    def __init__(self, z=None, x=None, sigma=0.2, resolution=100,
                 labels=None, fig_size=[8,8], cmap_type='jet',
                 interpolation_method='spline36'):
        # インプットが無効だった時のエラー処理
        if z is None:
            print('勝者位置をインプットして！！')
            exit(1)

        if x is None:
            print('データをインプットして！！')
            exit(1)

        # set input
        self.X = x
        self.Z = z
        self.Sigma = sigma
        self.Resolution = resolution
        self.K = resolution * resolution
        self.N = self.X.shape[0]
        self.L = self.Z.shape[1]

        # 描画キャンバスの設定
        self.Fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
        self.Map = self.Fig.add_subplot(1, 1, 1)
        self.Cmap_type = cmap_type
        self.labels = labels
        self.interpolation_method = interpolation_method

        # 潜在空間の代表点の設定
        self.Zeta = np.meshgrid(np.linspace(self.Z[:, 0].min(), self.Z[:, 0].max(), self.Resolution),
                           np.linspace(self.Z[:, 1].min(), self.Z[:, 1].max(), self.Resolution))
        self.Zeta = np.dstack(self.Zeta).reshape(self.K, self.L)


    # U-matrix表示
    def draw_umatrix(self):
        # U-matrix表示用の値を算出
        dY_std = self.__calc_umatrix()
        U_matrix_val = dY_std.reshape((self.Resolution, self.Resolution))

        # U-matrix表示
        self.Map.set_title("U-matrix")
        self.Map.imshow(U_matrix_val, interpolation=self.interpolation_method,
                      extent=[self.Zeta[:, 0].min(), self.Zeta[:, 0].max(),
                              self.Zeta[:, 1].max(), self.Zeta[:, 1].min()],
                   cmap=self.Cmap_type, vmax=0.5, vmin=-0.5)

        # ラベルの表示
        self.Map.scatter(x=self.Z[:, 0], y=self.Z[:, 1], c='k')
        umatrix_val = (np.random.rand(20, 20) - 0.5) * 2

        if self.labels is None:
            self.labels = np.arange(self.N) + 1

        # 勝者位置が重なった時用の処理
        epsilon = 0.04 * (self.Z.max() - self.Z.min())
        for i in range(self.N):
            count = 0
            for j in range(i):
                if np.allclose(self.Z[j, :], self.Z[i, :]):
                    count += 1
            self.Map.text(self.Z[i, 0], self.Z[i, 1] + epsilon * count, self.labels[i], color='k')
        plt.show()


    # U-matrix表示用の値（勾配）を算出
    def __calc_umatrix(self):
        # H, G, Rの算出
        dist_z = dist.cdist(self.Zeta, self.Z, 'sqeuclidean')
        H = np.exp(-dist_z / (2 * self.Sigma * self.Sigma))
        G = H.sum(axis=1)[:, np.newaxis]
        R = H / G

        # V, V_meanの算出
        V = R[:, :, np.newaxis] * (self.Z[np.newaxis, :, :] - self.Zeta[:, np.newaxis, :])          # KxNxL
        V_mean = V.sum(axis=1)[:, np.newaxis, :]                                                    # Kx1xL

        # dYdZの算出
        dRdZ = V - R[:, :, np.newaxis] * V_mean                                                     # KxNxL
        dYdZ = np.einsum("knl,nd->kld", dRdZ, self.X)     # KxLxD
        dYdZ_norm = np.sum(dYdZ ** 2, axis=(1, 2))                                                  # Kx1

        # 表示用の値を算出（標準化）
        sc = StandardScaler()
        dY_std = sc.fit_transform(dYdZ_norm[:, np.newaxis])


        return np.clip(dY_std / 4.0, -0.5, 0.5)
