import numpy as np
# import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from sklearn.preprocessing import StandardScaler
# import matplotlib.animation
from plotly.offline import plot
import plotly.graph_objs as go

class SOM_Umatrix:
    def __init__(self, X=None, Z=None, sigma=0.2, resolution=100,
                 labels=None, fig_size=[6,6], title_text='U-matrix', cmap_type='Jet',
                 interpolation_method='spline36', repeat=False, interval=40):
        # インプットが無効だった時のエラー処理
        if Z is None:
            raise ValueError('please input winner point')

        if X is None:
            raise ValueError('input observed data X')

        # set input
        self.X = X

        # set latent variables
        if Z.ndim == 3:
            # self.Z_allepoch = Z     # shape=(T,N,L)
            # self.isStillImage = False
            raise ValueError('3d ndarray is not supported yet')
        elif Z.ndim == 2:
            self.Z_allepoch = Z[None,:,:] # shape=(1,N,L)
            self.isStillImage = True

        # set sigma
        if isinstance(sigma, (int, float)):
            if self.isStillImage:
                self.sigma_allepoch = np.array([sigma,])
            else:
                raise ValueError("if Z is 3d array, sigma must be 1d array")
        elif sigma.ndim == 1:
            if self.isStillImage:
                raise ValueError("if Z is 2d array, sigma must be scalar")
            else:
                self.sigma_allepoch = sigma #shape=(T)
        else:
            raise ValueError("sigma must be 1d array or scalar")

        self.resolution = resolution

        self.N = self.X.shape[0]
        if self.N != self.Z_allepoch.shape[1]:
            raise ValueError("Z's sample number and X's one must be match")

        self.T = self.Z_allepoch.shape[0]

        self.L = self.Z_allepoch.shape[2]

        if self.L != 2:
            raise ValueError('latent variable must be 2dim')

        self.K = resolution ** self.L

        # 描画キャンバスの設定
        # self.Fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
        # self.Map = self.Fig.add_subplot(1, 1, 1)
        self.Cmap_type = cmap_type
        self.labels = labels
        if self.labels is None:
            self.labels = np.arange(self.N) + 1
        # self.interpolation_method = interpolation_method
        self.title_text = title_text
        # self.repeat = repeat
        # self.interval = interval

        # 潜在空間の代表点の設定
        self.Zeta = np.meshgrid(np.linspace(self.Z_allepoch[:, :, 0].min(), self.Z_allepoch[:, :, 0].max(), self.resolution),
                                np.linspace(self.Z_allepoch[:, :, 1].min(), self.Z_allepoch[:, :, 1].max(), self.resolution))
        self.Zeta = np.dstack(self.Zeta).reshape(self.K, self.L)


    # U-matrix表示
    def draw_umatrix(self):

        # U-matrixの初期状態を表示する
        Z = self.Z_allepoch[0,:,:]
        sigma = self.sigma_allepoch[0]

        # U-matrix表示用の値を算出
        #dY_std = self.__calc_umatrix(Z,sigma)
        #U_matrix_val = dY_std.reshape((self.resolution, self.resolution))
        U_matrix_val = self.__calc_umatrix(Z,sigma)

        # U-matrix表示
        # self.Map.set_title(self.title_text)
        # self.Im = self.Map.imshow(U_matrix_val, interpolation=self.interpolation_method,
        #               extent=[self.Zeta[:, 0].min(), self.Zeta[:, 0].max(),
        #                       self.Zeta[:, 1].max(), self.Zeta[:, 1].min()],
        #            cmap=self.Cmap_type, vmax=0.5, vmin=-0.5, animated=True)

        trace_scat = go.Scatter(x=Z[:,0],y=Z[:,1],
                                mode='markers+text',
                                text=self.labels,
                                textposition='bottom center')
        trace_umatrix = go.Heatmap(x=self.Zeta[:,0],y=self.Zeta[:,1],
                                   z=U_matrix_val,colorscale=self.Cmap_type,
                                   zsmooth='best')
        # ラベルの表示
        # self.Scat = self.Map.scatter(x=Z[:, 0], y=Z[:, 1], c='k')
        layout = go.Layout(
            width = 800,
            height = 800,
            showlegend=False,
            title = self.title_text
        )
        data = [trace_umatrix,trace_scat]
        fig = go.Figure(data=data,layout=layout)
        plot(fig)



        # 勝者位置が重なった時用の処理
        # self.Label_Texts = []
        # self.epsilon = 0.04 * (self.Z_allepoch.max() - self.Z_allepoch.min())
        # for i in range(self.N):
        #     count = 0
        #     for j in range(i):
        #         if np.allclose(Z[j, :], Z[i, :]):
        #             count += 1
        #     Label_Text = self.Map.text(Z[i, 0], Z[i, 1] + self.epsilon * count, self.labels[i], color='k')
        #     self.Label_Texts.append(Label_Text)

        # ani = matplotlib.animation.FuncAnimation(self.Fig, self.update, interval=self.interval, blit=False,
        #                                    repeat=self.repeat, frames=self.T)
        # plt.show()

    # def update(self, epoch):
    #     Z = self.Z_allepoch[epoch,:,:]
    #     sigma = self.sigma_allepoch[epoch]
    #
    #     dY_std = self.__calc_umatrix(Z, sigma)
    #     U_matrix_val = dY_std.reshape((self.resolution, self.resolution))
    #     self.Im.set_array(U_matrix_val)
    #     self.Scat.set_offsets(Z)
    #     for i in range(self.N):
    #         count = 0
    #         for j in range(i):
    #             if np.allclose(Z[j, :], Z[i, :]):
    #                 count += 1
    #         self.Label_Texts[i].remove()
    #         self.Label_Texts[i] = self.Map.text(Z[i, 0],
    #                                             Z[i, 1] + self.epsilon * count,
    #                                             self.labels[i],
    #                                             color='k')
    #     if not self.isStillImage:
    #         self.Map.set_title(self.title_text+' epoch={}'.format(epoch))

    # U-matrix表示用の値（勾配）を算出
    def __calc_umatrix(self, Z, sigma):
        # H, G, Rの算出
        dist_z = dist.cdist(self.Zeta, Z, 'sqeuclidean')
        H = np.exp(-dist_z / (2 * sigma * sigma))
        G = H.sum(axis=1)[:, np.newaxis]
        R = H / G

        # V, V_meanの算出
        V = R[:, :, np.newaxis] * (Z[np.newaxis, :, :] - self.Zeta[:, np.newaxis, :])          # KxNxL
        V_mean = V.sum(axis=1)[:, np.newaxis, :]                                                    # Kx1xL

        # dYdZの算出
        dRdZ = V - R[:, :, np.newaxis] * V_mean                                                     # KxNxL
        dYdZ = np.einsum("knl,nd->kld", dRdZ, self.X)     # KxLxD
        dYdZ_norm = np.sum(dYdZ ** 2, axis=(1, 2))                                                  # Kx1

        # 表示用の値を算出（標準化）
        sc = StandardScaler()
        dY_std = sc.fit_transform(dYdZ_norm[:, np.newaxis])
        # dY_std = sc.fit_transform(dYdZ_norm[])


        return np.clip(dY_std.ravel() / 4.0, -0.5, 0.5)
