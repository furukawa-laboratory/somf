import numpy as np
import scipy.spatial.distance as dist
from tqdm import tqdm


class UnsupervisedKernelRegression(object):
    def __init__(self, X, n_components, bandwidth_gaussian_kernel=1.0,
                 is_compact=False, lambda_=1.0,
                 init='random', is_loocv=False, is_save_history=False):
        self.X = X.copy()
        self.n_samples = X.shape[0]
        self.n_dimensions = X.shape[1]
        self.n_components = n_components
        self.bandwidth_gaussian_kernel = bandwidth_gaussian_kernel
        self.precision = 1.0 / (bandwidth_gaussian_kernel * bandwidth_gaussian_kernel)
        self.is_compact = is_compact
        self.is_loocv = is_loocv
        self.is_save_hisotry = is_save_history

        self.Z = None
        if isinstance(init, str) and init in 'random':
            self.Z = np.random.normal(0, 1.0, (self.n_samples, self.n_components)) * bandwidth_gaussian_kernel * 0.5
        elif isinstance(init, np.ndarray) and init.shape == (self.n_samples, self.n_components):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        self.lambda_ = lambda_

        self._done_fit = False

    def fit(self, nb_epoch=100, verbose=True, eta=0.5, expand_epoch=None):

        K = self.X @ self.X.T

        self.nb_epoch = nb_epoch

        if self.is_save_hisotry:
            self.history = {}
            self.history['z'] = np.zeros((nb_epoch, self.n_samples, self.n_components))
            self.history['y'] = np.zeros((nb_epoch, self.n_samples, self.n_dimensions))
            self.history['zvar'] = np.zeros((nb_epoch, self.n_components))
            self.history['obj_func'] = np.zeros(nb_epoch)

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        for epoch in bar:
            Delta = self.Z[:, None, :] - self.Z[None, :, :]
            DistZ = np.sum(np.square(Delta), axis=2)
            H = np.exp(-0.5 * self.precision * DistZ)
            if self.is_loocv:
                H -= np.identity(H.shape[0])

            G = np.sum(H, axis=1)[:, None]
            GInv = 1 / G
            R = H * GInv

            Y = R @ self.X
            DeltaYX = Y[:, None, :] - self.X[None, :, :]
            Error = Y - self.X
            obj_func = np.sum(np.square(Error)) / self.n_samples + self.lambda_ * np.sum(np.square(self.Z))

            A = self.precision * R * np.einsum('nd,nid->ni', Y - self.X, DeltaYX)
            dFdZ = -2.0 * np.sum((A + A.T)[:, :, None] * Delta, axis=1) / self.n_samples

            dFdZ -= 2.0 * self.lambda_ * self.Z

            self.Z += eta * dFdZ
            if self.is_compact:
                self.Z = np.clip(self.Z, -1.0, 1.0)
            else:
                self.Z -= self.Z.mean(axis=0)

            if self.is_save_hisotry:
                self.history['z'][epoch] = self.Z
                self.history['obj_func'][epoch] = obj_func

        self._done_fit = True
        return self.history

    def calculate_history_of_mapping(self, resolution, size='auto'):
        """
        :param resolution:
        :param size:
        :return:
        """
        if not self._done_fit:
            raise ValueError("fit is not done")

        self.resolution = resolution
        Zeta = create_zeta(-1, 1, self.n_components, resolution)
        M = Zeta.shape[0]

        self.history['f'] = np.zeros((self.nb_epoch, M, self.n_dimensions))

        for epoch in range(self.nb_epoch):
            Z = self.history['z'][epoch]
            if size == 'auto':
                Zeta = create_zeta(Z.min(), Z.max(), self.n_components, resolution)
            else:
                Zeta = create_zeta(size.min(), size.max(), self.n_components, resolution)

            Dist = dist.cdist(Zeta, Z, 'sqeuclidean')

            H = np.exp(-0.5 * self.precision * Dist)
            G = np.sum(H, axis=1)[:, None]
            GInv = np.reciprocal(G)
            R = H * GInv

            Y = np.dot(R, self.X)

            self.history['f'][epoch] = Y

    def transform(self, Xnew, nb_epoch_trans=100, eta_trans=0.5, verbose=True, constrained=True):
        # calculate latent variables of new data using gradient descent
        # objective function is square error E = ||f(z)-x||^2

        if not self._done_fit:
            raise ValueError("fit is not done")

        Nnew = Xnew.shape[0]

        # initialize Znew, using latent variables of observed data
        Dist_Xnew_X = dist.cdist(Xnew, self.X)
        BMS = np.argmin(Dist_Xnew_X, axis=1)  # calculate Best Matching Sample
        Znew = self.Z[BMS, :]  # initialize Znew

        if verbose:
            bar = tqdm(range(nb_epoch_trans))
        else:
            bar = range(nb_epoch_trans)

        for epoch in bar:
            # calculate gradient
            Delta = self.Z[None, :, :] - Znew[:, None, :]  # shape = (Nnew,N,L)
            Dist_Znew_Z = dist.cdist(Znew, self.Z, "sqeuclidean")  # shape = (Nnew,N)
            H = np.exp(-0.5 * self.precision * Dist_Znew_Z)  # shape = (Nnew,N)
            G = np.sum(H, axis=1)[:, None]  # shape = (Nnew,1)
            Ginv = np.reciprocal(G)  # shape = (Nnew,1)
            R = H * Ginv  # shape = (Nnew,N)
            F = R @ self.X  # shape = (Nnew,D)

            Delta_bar = np.einsum("kn,knl->kl", R, Delta)  # (Nnew,N)times(Nnew,N,L)=(Nnew,L)
            # Delta_bar = np.sum(R[:,:,None] * Delta, axis=1)           # same calculate
            dRdZ = self.precision * R[:, :, None] * (Delta - Delta_bar[:, None, :])  # shape = (Nnew,N,L)

            dFdZ = np.einsum("nd,knl->kdl", self.X, dRdZ)  # shape = (Nnew,D,L)
            # dFdZ = np.sum(self.X[None,:,:,None]*dRdZ[:,:,None,:],axis=1)  # same calculate
            dEdZ = 2.0 * np.einsum("kd,kdl->kl", F - Xnew, dFdZ)  # shape (Nnew, L)
            # update latent variables
            Znew -= eta_trans * dEdZ
            if self.is_compact:
                Znew = np.clip(Znew, -1.0, 1.0)
            if constrained:
                Znew = np.clip(Znew, self.Z.min(axis=0), self.Z.max(axis=0))

        return Znew

    def inverse_transform(self, Znew):
        if not self._done_fit:
            raise ValueError("fit is not done")
        if Znew.shape[1] != self.n_components:
            raise ValueError("Znew dimension must be {}".format(self.n_components))

        Dist_Znew_Z = dist.cdist(Znew, self.Z, "sqeuclidean")  # shape = (Nnew,N)
        H = np.exp(-0.5 * self.precision * Dist_Znew_Z)  # shape = (Nnew,N)
        G = np.sum(H, axis=1)[:, None]  # shape = (Nnew,1)
        Ginv = np.reciprocal(G)  # shape = (Nnew,1)
        R = H * Ginv  # shape = (Nnew,N)
        F = R @ self.X  # shape = (Nnew,D)

        return F

    def visualize(self, n_grid_points=30, label_data=None, label_feature=None, fig_size=None):
        # invalid check
        if self.n_components != 2:
            raise ValueError('Now support only n_components = 2')

        # import necessary library to draw
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        self._initialize_to_visualize(n_grid_points, label_data, label_feature, fig_size)

        self._draw_latent_space()
        self._draw_features()

        # connect figure and method defining action when latent space is clicked
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)
        plt.show()

    def _initialize_to_visualize(self, n_grid_points, label_data, label_feature, fig_size):
        import matplotlib.pyplot as plt
        if isinstance(n_grid_points, int):
            # 代表点の数を潜在空間の次元ごとに格納
            self.n_grid_points = np.ones(self.n_components, dtype='int8') * n_grid_points
        else:
            raise ValueError('Only support n_grid_points is int')
        if self.is_compact:
            self.grid_points = create_zeta(-1.0, 1.0, self.n_components, n_grid_points)
        else:
            raise ValueError('Not support is_compact=False')  # create_zetaの整備が必要なので実装は後で
        self.click_point_latent_space = 0  # index of the clicked representative point
        self.clicked_mapping = self.X.mean(axis=0)
        self.is_initial_view = True
        self.selected_feature = None

        self.grid_mapping = self.inverse_transform(self.grid_points)
        # invalid check
        if label_data is None:
            self.label_data = label_data
        elif isinstance(label_data, list):
            self.label_data = label_data
        elif isinstance(label_data, np.ndarray):
            if np.squeeze(label_data).ndim == 1:
                self.label_data = np.squeeze(label_data)
            else:
                raise ValueError('label_data must be 1d array')
        else:
            raise ValueError('label_data must be 1d array or list')

        if label_feature is None:
            self.label_feature = np.arange(self.n_dimensions)
        elif isinstance(label_feature, list):
            self.label_feature = label_feature
        elif isinstance(label_feature, np.ndarray):
            if np.squeeze(label_feature).ndim == 1:
                self.label_feature = np.squeeze(label_feature)
            else:
                raise ValueError('label_feature must be 1d array')
        else:
            raise ValueError('label_feature must be 1d array or list')

        if fig_size is None:
            self.fig = plt.figure(figsize=(15, 6))
        else:
            self.fig = plt.figure(figsize=fig_size)
        self.ax_latent_space = self.fig.add_subplot(1, 2, 1, aspect='equal')
        self.ax_latent_space.set_title('Latent space')
        self.ax_features = self.fig.add_subplot(1, 2, 2)
        self.ax_features.set_title('Mean of mapping')

        epsilon = 0.03 * np.abs(self.grid_points.max() - self.grid_points.min())
        self.noise_label = epsilon * (np.random.rand(self.n_samples, self.n_components) * 2.0 - 1.0)

    def _draw_latent_space(self):
        self.ax_latent_space.cla()
        if self.selected_feature is not None:
            values_selected_feature = self.grid_mapping[:, self.selected_feature]
            values_selected_feature_2d = self.__unflatten_grid_array(values_selected_feature)
            representative_points_2d = self.__unflatten_grid_array(self.grid_points)
            pcm = self.ax_latent_space.pcolormesh(representative_points_2d[:, :, 0],
                                                  representative_points_2d[:, :, 1],
                                                  values_selected_feature_2d)
            ctr = self.ax_latent_space.contour(representative_points_2d[:, :, 0],
                                               representative_points_2d[:, :, 1],
                                               values_selected_feature_2d, 6, colors='k')
            self.ax_latent_space.clabel(ctr)
        self.ax_latent_space.scatter(self.Z[:, 0], self.Z[:, 1])
        if self.label_data is None:
            pass
        else:
            for z, noise, label in zip(self.Z, self.noise_label, self.label_data):
                point_label = z + noise
                self.ax_latent_space.text(point_label[0], point_label[1], label,
                                          ha='center', va='bottom', color='black')
        if self.is_initial_view:
            pass
        else:
            self.__draw_click_point_latent_space()
        self.fig.show()

    def _draw_features(self):
        self.ax_features.cla()
        self.feature_bars = self.ax_features.bar(self.label_feature, self.clicked_mapping)
        if self.selected_feature is not None:
            self.feature_bars[self.selected_feature].set_color('r')
        self.ax_features.set_ylim(self.X.min(), self.X.max() * 1.05)
        if self.is_initial_view:
            self.ax_features.set_title('mean of data')
        else:
            self.ax_features.set_title('Features')
        self.ax_features.set_xticklabels(labels=self.label_feature, rotation=270)
        self.fig.show()

    def __draw_click_point_latent_space(self):
        coordinate = self.grid_points[self.click_point_latent_space]
        self.ax_latent_space.plot(coordinate[0], coordinate[1],
                                  ".", color="red", ms=20, fillstyle="none")

    def __onclick_fig(self, event):
        self.is_initial_view = False
        if event.xdata is not None:
            if event.inaxes == self.ax_latent_space.axes:  # 潜在空間をクリックしたかどうか
                # クリックされた座標の取得
                click_coordinates = np.array([event.xdata, event.ydata])

                # クリックしたところといちばん近い代表点がどこかを計算
                self.click_point_latent_space = self.__calc_nearest_representative_point(click_coordinates)

                # その代表点の写像先の特徴量を計算
                self.__calc_features()

                # その特徴量の値を描画
                self._draw_latent_space()
                self._draw_features()
            elif event.inaxes == self.ax_features.axes:  # map2がクリックされた時
                click_coordinates = np.array([event.xdata, event.ydata])
                for i, bar in enumerate(self.feature_bars):
                    if click_coordinates[0] > bar._x0 and click_coordinates[0] < bar._x1:
                        self.selected_feature = i
                        self._draw_latent_space()
                        self._draw_features()

    def __calc_nearest_representative_point(self, click_point):
        distance = dist.cdist(self.grid_points, click_point.reshape(1, -1))
        index_nearest = np.argmin(distance)
        return index_nearest

    def __calc_features(self):
        self.clicked_mapping = self.grid_mapping[self.click_point_latent_space, :]

    def __unflatten_grid_array(self, grid_array):
        if grid_array.shape[0] == np.prod(self.n_grid_points):
            return np.squeeze(grid_array.reshape(np.append(self.n_grid_points, -1)))
        else:
            raise ValueError('arg shape {} is not consistent'.format(grid_array.shape))


def create_zeta(zeta_min, zeta_max, latent_dim, resolution):
    mesh1d, step = np.linspace(zeta_min, zeta_max, resolution, endpoint=False, retstep=True)
    mesh1d += step / 2.0
    if latent_dim == 1:
        Zeta = mesh1d
    elif latent_dim == 2:
        Zeta = np.meshgrid(mesh1d, mesh1d)
    else:
        raise ValueError("invalid latent dim {}".format(latent_dim))
    Zeta = np.dstack(Zeta).reshape(-1, latent_dim)
    return Zeta
