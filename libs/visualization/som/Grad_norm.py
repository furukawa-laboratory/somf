import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from sklearn.preprocessing import StandardScaler
import matplotlib.animation
from ...tools.calc_grad_norm_of_ks import calc_grad_norm_of_ks as calc_grad_norm


class Grad_Norm:
    def __init__(self, X=None, Z=None, sigma=0.2, resolution=100,
                 labels=None, fig_size=[6, 6], title_text='Grad_norm', cmap_type='jet',
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
            self.Z_allepoch = Z  # shape=(T,N,L)
            self.isStillImage = False
        elif Z.ndim == 2:
            self.Z_allepoch = Z[None, :, :]  # shape=(1,N,L)
            self.isStillImage = True

        # set sigma
        if isinstance(sigma, (int, float)):
            if self.isStillImage:
                self.sigma_allepoch = np.array([sigma, ])
            else:
                raise ValueError("if Z is 3d array, sigma must be 1d array")
        elif sigma.ndim == 1:
            if self.isStillImage:
                raise ValueError("if Z is 2d array, sigma must be scalar")
            else:
                self.sigma_allepoch = sigma  # shape=(T)
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
        self.Fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
        self.Map = self.Fig.add_subplot(1, 1, 1)
        self.Cmap_type = cmap_type
        self.labels = labels
        if self.labels is None:
            self.labels = np.arange(self.N) + 1
        self.interpolation_method = interpolation_method
        self.title_text = title_text
        self.repeat = repeat
        self.interval = interval

        # 潜在空間の代表点の設定
        self.Zeta = np.meshgrid(
            np.linspace(self.Z_allepoch[:, :, 0].min(), self.Z_allepoch[:, :, 0].max(), self.resolution),
            np.linspace(self.Z_allepoch[:, :, 1].min(), self.Z_allepoch[:, :, 1].max(), self.resolution))
        self.Zeta = np.dstack(self.Zeta).reshape(self.K, self.L)

    # Grad_norm表示
    def draw_umatrix(self):

        # Grad_normの初期状態を表示する
        Z = self.Z_allepoch[0, :, :]
        sigma = self.sigma_allepoch[0]

        # Grad_norm表示用の値を算出
        dY_std = calc_grad_norm(Zeta=self.Zeta, Z=Z, X=self.X, sigma=sigma) # return value in [-2.0,2.0]
        U_matrix_val = dY_std.reshape((self.resolution, self.resolution))

        # Grad_norm表示
        self.Map.set_title(self.title_text)
        self.Im = self.Map.imshow(U_matrix_val, interpolation=self.interpolation_method,
                                  extent=[self.Zeta[:, 0].min(), self.Zeta[:, 0].max(),
                                          self.Zeta[:, 1].max(), self.Zeta[:, 1].min()],
                                  cmap=self.Cmap_type, vmax=2.0, vmin=-2.0, animated=True)

        # ラベルの表示
        self.Scat = self.Map.scatter(x=Z[:, 0], y=Z[:, 1], c='k')

        # 勝者位置が重なった時用の処理
        self.Label_Texts = []
        self.epsilon = 0.04 * (self.Z_allepoch.max() - self.Z_allepoch.min())
        for i in range(self.N):
            count = 0
            for j in range(i):
                if np.allclose(Z[j, :], Z[i, :]):
                    count += 1
            Label_Text = self.Map.text(Z[i, 0], Z[i, 1] + self.epsilon * count, self.labels[i], color='k')
            self.Label_Texts.append(Label_Text)

        ani = matplotlib.animation.FuncAnimation(self.Fig, self.update, interval=self.interval, blit=False,
                                                 repeat=self.repeat, frames=self.T)
        plt.show()

    def update(self, epoch):
        Z = self.Z_allepoch[epoch, :, :]
        sigma = self.sigma_allepoch[epoch]

        dY_std = calc_grad_norm(Zeta=self.Zeta, Z=Z, X=self.X, sigma=sigma) # return value in [-2.0,2.0]
        U_matrix_val = dY_std.reshape((self.resolution, self.resolution))
        self.Im.set_array(U_matrix_val)
        self.Scat.set_offsets(Z)
        for i in range(self.N):
            count = 0
            for j in range(i):
                if np.allclose(Z[j, :], Z[i, :]):
                    count += 1
            self.Label_Texts[i].remove()
            self.Label_Texts[i] = self.Map.text(Z[i, 0],
                                                Z[i, 1] + self.epsilon * count,
                                                self.labels[i],
                                                color='k')
        if not self.isStillImage:
            self.Map.set_title(self.title_text + ' epoch={}'.format(epoch))

