import numpy as np
import scipy.spatial.distance as dist
from tqdm import tqdm
from ..tools.create_zeta import create_zeta


class UnsupervisedKernelRegression(object):
    def __init__(self, X, n_components, bandwidth_gaussian_kernel=1.0,
                 is_compact=False, lambda_=1.0, weights=None,
                 init='random', is_loocv=False, is_save_history=False):
        self.X = X.copy()
        self.n_samples = X.shape[0]
        self.n_dimensions = X.shape[1]
        self.n_components = n_components
        self.bandwidth_gaussian_kernel = bandwidth_gaussian_kernel
        self.precision = 1.0 / (bandwidth_gaussian_kernel * bandwidth_gaussian_kernel)
        self.is_compact = is_compact
        self.is_loocv = is_loocv
        self.is_save_history = is_save_history

        self.Z = None
        if isinstance(init, str) and init in 'random':
            self.Z = np.random.normal(0, 1.0, (self.n_samples, self.n_components)) * bandwidth_gaussian_kernel * 0.5
        elif isinstance(init, np.ndarray) and init.shape == (self.n_samples, self.n_components):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        self.lambda_ = lambda_

        if weights is None:
            self.weights = np.ones(self.n_samples)
        elif weights.shape[0] == self.n_samples and np.squeeze(weights).ndim == 1:
            if np.all(weights >= 0):
                self.weights = np.squeeze(weights)
            else:
                raise ValueError("weight don't include non-negative value")
        else:
            raise ValueError('weight shape must match X shape')

        self._done_fit = False

    def fit(self, nb_epoch=100, verbose=True, eta=0.5, expand_epoch=None):

        self.nb_epoch = nb_epoch

        if self.is_save_history:
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
            H = H * self.weights[None, :]

            G = np.sum(H, axis=1)[:, None]
            GInv = 1 / G
            R = H * GInv

            Y = R @ self.X
            Error = Y - self.X
            obj_func = (np.sum(np.square(Error) * self.weights[:, None]) / self.weights.sum()
                        + self.lambda_ * np.sum(np.square(self.Z))
                        )
            product_delta = (np.einsum('nd,nd->n', Y, Y)[:, None]
                             - np.einsum('nd,id->ni', Y, self.X)
                             - np.einsum('nd,nd->n', self.X, Y)[:, None]
                             + np.einsum('nd,id->ni', self.X, self.X)
                             )

            A = np.einsum('ni,n,ni->ni', R, self.weights, product_delta)
            dFdZ = 2.0 * self.precision * np.einsum('ni,nil->nl', A + A.T, Delta) / self.weights.sum()

            dFdZ += 2.0 * self.lambda_ * self.Z

            self.Z -= eta * dFdZ
            if self.is_compact:
                self.Z = np.clip(self.Z, -1.0, 1.0)
            else:
                self.Z -= self.Z.mean(axis=0)

            if self.is_save_history:
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

            H = np.exp(-0.5 * self.precision * Dist) * self.weights[None, :]
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
            H = np.exp(-0.5 * self.precision * Dist_Znew_Z) * self.weights[None, :]  # shape = (Nnew,N)
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
        H = np.exp(-0.5 * self.precision * Dist_Znew_Z) * self.weights[None, :]  # shape = (Nnew,N)
        G = np.sum(H, axis=1)[:, None]  # shape = (Nnew,1)
        Ginv = np.reciprocal(G)  # shape = (Nnew,1)
        R = H * Ginv  # shape = (Nnew,N)
        F = R @ self.X  # shape = (Nnew,D)

        return F

    def visualize(self, n_grid_points=30, label_data=None, label_feature=None,
                  marker='.', is_show_all_label_data=False, is_middle_color_zero=False,
                  params_imshow=None, params_scatter=None,
                  title_latent_space=None, title_feature_bars=None,
                  fig=None, fig_size=None, ax_latent_space=None, ax_feature_bars=None):
        """Visualize fit model interactively.
        The dataset can be visualized in an exploratory way using the latent variables and the mapping estimated by UKR.
        When an arbitrary coordinate on the latent space is specified, the corresponding feature is displayed as a bar.
        Also, if a bar of a particular feature is specified, the latent space is colored by its value.

        Note: This method uses TkAgg as a backend. Therefore, if matplotlib is imported beforehand
        and a different backend is specified, it will not work.

        Parameters
        ----------
        n_grid_points: int, optional, default = None
            Number of representative points of discretization of the latent space needed for the drawing.
        label_data: array of shape (n_data, ), optional. default = None
            The labels corresponds rows of the dataset X.
        label_feature: array of shape (n_features, ), optional. default = None
            The labels corresponds columns of the dataset X.
        marker: MarkerStyle or its sequence, optional, default = None
            The marker of scatter. It allows multiple markers.
        is_show_all_label_data: bool, optional, default = False
            When True the labels of the data is always shown.
            When False the label is only shown when the cursor overlaps the corresponding latent variable.
        is_middle_color_zero: bool, optional, default = False
            If `True`, the value corresponding to middle color in the colormap is fixed at 0.
            If the data is normalized to zero mean and it is important whether it is higher or lower than the mean,
            setting to `True` makes the coloring easier to read.
        params_imshow: dict, optional, default = None
            The dictionary of kwargs in imshow used for drawing the value of the feature in the latent space.
            See the official document of matplotlib.pyplot.imshow for more information on available arguments.
        params_scatter: dict, optional, default = None
            The dictionary of kwargs in scatter used for plotting latent variables in the latent space.
            See the official document of matplotlib.pyplot.imshow for more information on available arguments.
        title_latent_space: str, optional, default = None
            The title of axis to visualize the latent space
        title_feature_bars: str, optional, default = None
            The title of axis to visualize bars of features
        fig: matplotlib.figure.Figure, default = True
            The figure to visualize.
            It is assigned only when you want to specify a figure to visualize.
        fig_size: (float, float), optional, default = None
            The size of figure.
        ax_latent_space: matplotlib.axes._subplots.AxesSubplot, optional, default = False
            The axis to visualize latent space.
        ax_feature_bars: matplotlib.axes._subplots.AxesSubplot, optional, default = False
            The axis to visualize feature_bars.

        Returns
        -------
        No returns
        """

        # import library to draw
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        self._initialize_to_visualize(n_grid_points=n_grid_points,
                                      params_imshow=params_imshow,
                                      params_scatter=params_scatter,
                                      label_data=label_data,
                                      label_feature=label_feature,
                                      title_latent_space=title_latent_space,
                                      title_feature_bars=title_feature_bars,
                                      is_show_all_label_data=is_show_all_label_data,
                                      is_middle_color_zero=is_middle_color_zero,
                                      fig=fig,
                                      fig_size=fig_size,
                                      ax_latent_space=ax_latent_space,
                                      ax_feature_bars=ax_feature_bars,
                                      marker=marker)

        self._draw_latent_space()
        self._draw_feature_bars()

        # connect figure and method defining action when latent space is clicked
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)
        # connect figure and method defining action when mouse over
        if self.label_data is not None and self.is_show_all_label_data is False:
            self.fig.canvas.mpl_connect('motion_notify_event', self.__mouse_over_fig)
            # self.fig.canvas.mpl_connect('axes_leave_event', self.__mouse_leave_fig)
        plt.show()

    def __onclick_fig(self, event):
        self.is_initial_view = False
        if event.xdata is not None:
            # クリックされた座標の取得
            click_coordinates = np.array([event.xdata, event.ydata])
            if event.inaxes == self.ax_latent_space.axes:  # 潜在空間をクリックしたかどうか
                self._set_feature_bar_from_latent_space(click_coordinates)
                self._draw_latent_space()
                self._draw_feature_bars()
            elif event.inaxes == self.ax_feature_bars.axes:  # 特徴量のバーがクリックされたかどうか
                self._set_latent_space_from_feature_bar(click_coordinates)
                self._draw_latent_space()
                self._draw_feature_bars()

    def __mouse_over_fig(self, event):
        if event.xdata is not None:
            # クリックされた座標の取得
            over_coordinates = np.array([event.xdata, event.ydata])
            if event.inaxes == self.ax_latent_space.axes:
                self._set_shown_label_in_latent_space(over_coordinates)
                self._draw_latent_space()
                self._draw_feature_bars()

            elif event.inaxes == self.ax_feature_bars:
                pass

    def _initialize_to_visualize(self, n_grid_points, label_data, label_feature,
                                 marker, is_show_all_label_data, is_middle_color_zero,
                                 params_imshow, params_scatter,
                                 title_latent_space, title_feature_bars,
                                 fig, fig_size, ax_latent_space, ax_feature_bars):

        # invalid check
        if self.n_components != 2:
            raise ValueError('Now support only n_components = 2')

        if isinstance(n_grid_points, int):
            # 代表点の数を潜在空間の次元ごとに格納
            self.n_grid_points = np.ones(self.n_components, dtype='int8') * n_grid_points
            if self.is_compact:
                self._set_grid(create_zeta(-1.0, 1.0, self.n_components, n_grid_points), self.n_grid_points)
            else:
                raise ValueError('Not support is_compact=False')  # create_zetaの整備が必要なので実装は後で
        else:
            raise ValueError('Only support n_grid_points is int')

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


        if params_imshow is None:
            self.params_imshow = {}
        elif isinstance(params_imshow, dict):
            self.params_imshow = params_imshow
        else:
            raise ValueError('invalid params_imshow={}'.format(params_imshow))

        if params_scatter is None:
            self.params_scatter = {}
            self.params_scatter['s'] = 10
        elif isinstance(params_scatter, dict):
            self.params_scatter = params_scatter
        else:
            raise ValueError('invalid params_scatter={}'.format(params_scatter))

        if isinstance(marker, str):
            self.params_scatter['marker'] = marker
            self.multiple_marker = None
        elif isinstance(marker, (list, tuple, np.ndarray)):
            self.multiple_marker = np.array(marker)
        else:
            raise ValueError('invalid marker={}'.format(marker))

        if title_latent_space is None:
            self.title_latent_space = 'Latent space'
        else:
            self.title_latent_space = title_latent_space

        if title_feature_bars is None:
            self.title_feature_bars = 'Feature bars'
        else:
            self.title_feature_bars = title_feature_bars

        self.is_show_all_label_data = is_show_all_label_data

        # set variables
        if fig is None:
            import matplotlib.pyplot as plt
            if fig_size is None:
                self.fig = plt.figure(figsize=(15, 6))
            else:
                self.fig = plt.figure(figsize=fig_size)
        else:
            self.fig = fig

        if ax_latent_space is None and ax_feature_bars is None:
            self.ax_latent_space = self.fig.add_subplot(1, 2, 1, aspect='equal')
            self.ax_feature_bars = self.fig.add_subplot(1, 2, 2)
        else:
            self.ax_latent_space = ax_latent_space
            self.ax_feature_bars = ax_feature_bars

        self.is_middle_color_zero = is_middle_color_zero
        self.click_point_latent_space = None  # index of the clicked representative point
        self.clicked_mapping = self.X.mean(axis=0)
        self.is_initial_view = True
        self.selected_feature = None
        self.grid_values_to_draw = None
        self.index_data_label_shown = None
        self.mask_latent_variables = np.full(self.n_samples, True, bool)

    def _set_grid(self, grid_points, n_grid_points):
        self.grid_points = grid_points
        self.grid_mapping = self.inverse_transform(self.grid_points)
        if grid_points.shape[0] == np.prod(n_grid_points):
            self.n_grid_points = n_grid_points
        else:
            raise ValueError('grid_points shape and n_grid_points one are not consistent')
        self.threshold_radius_show = np.abs(self.grid_points.max() - self.grid_points.min()) * 0.05
        epsilon = 0.03 * np.abs(self.grid_points.max() - self.grid_points.min())
        self.noise_label = epsilon * (np.random.rand(self.n_samples, self.n_components) * 2.0 - 1.0)

    def _set_feature_bar_from_latent_space(self, click_coordinates):
        if self.click_point_latent_space is not None:
            previous_coordinate = self.grid_points[self.click_point_latent_space]
            dist = np.sqrt(np.sum(np.square(previous_coordinate - click_coordinates)))
            epsilon = 0.02 * np.abs(self.grid_points.max() - self.grid_points.min())
            if dist < epsilon:
                is_uncondition = True
            else:
                is_uncondition = False
        else:
            is_uncondition = False

        if is_uncondition:
            self.click_point_latent_space = None
            self.clicked_mapping = self.X.mean(axis=0)
        else:
            # クリックしたところといちばん近い代表点がどこかを計算
            self.click_point_latent_space = self.__calc_nearest_grid_point(click_coordinates)

            # その代表点の写像先の特徴量を計算
            self.clicked_mapping = self.grid_mapping[self.click_point_latent_space, :]

    def _set_latent_space_from_feature_bar(self, click_coordinates):
        for i, bar in enumerate(self.feature_bars):
            if click_coordinates[0] > bar._x0 and click_coordinates[0] < bar._x1:
                self.selected_feature = i
                self.grid_values_to_draw = self.grid_mapping[:, i]

    def _set_shown_label_in_latent_space(self, click_coordinates):
        index, dist = self.__calc_nearest_latent_variable(click_coordinates, retdist=True)
        if dist <= self.threshold_radius_show:
            self.index_data_label_shown = index
        else:
            self.index_data_label_shown = None

    def _set_grid_values_to_draw(self, grid_values):
        self.grid_values_to_draw = grid_values
        self.selected_feature = None

    def set_params_imshow(self, params: dict):
        if isinstance(params, dict):
            self.params_imshow.update(params)
        else:
            raise ValueError('invalid params={}')

    def set_params_scatter(self, params: dict):
        if isinstance(params, dict):
            self.params_scatter.update(params)
        else:
            raise ValueError('invalid params={}')

    def set_mask_latent_variables(self, mask):
        self.mask_latent_variables = mask

    def _set_titles(self, title_latent_space, title_feature_bars):
        self.title_latent_space = title_latent_space
        self.title_feature_bars = title_feature_bars

    def _draw_latent_space(self):
        import matplotlib.pyplot as plt
        from matplotlib import patheffects as path_effects
        self.ax_latent_space.cla()
        self.ax_latent_space.set_title(self.title_latent_space)
        self.ax_latent_space.set_xlim(self.Z[:, 0].min() * 1.05, self.Z[:, 0].max() * 1.05)
        self.ax_latent_space.set_ylim(self.Z[:, 1].min() * 1.05, self.Z[:, 1].max() * 1.05)

        # Draw color using self.grid_values_to_draw by pcolormesh and contour
        if self.grid_values_to_draw is not None:
            # set vmin and vmax
            if self.is_middle_color_zero:
                max_grid_value = self.grid_values_to_draw.max()
                min_grid_value = self.grid_values_to_draw.min()
                vmin = -max(abs(max_grid_value), abs(min_grid_value))
                vmax = max(abs(max_grid_value), abs(min_grid_value))
            else:
                vmin = None
                vmax = None

            # To draw by imshow and contour, reshape arrays like grid
            grid_values_to_imshow = self.__unflatten_grid_array(self.grid_values_to_draw)
            grid_values_to_contour = self.__unflatten_grid_array(self.grid_values_to_draw)
            grid_points_3d = self.__unflatten_grid_array(self.grid_points)

            # set coordinate of axis
            any_index = 0
            if grid_points_3d[any_index, 0, 0] < grid_points_3d[any_index, -1, 0]:
                coordinate_ax_left = grid_points_3d[any_index, 0, 0]
                coordinate_ax_right = grid_points_3d[any_index, -1, 0]
            else:
                coordinate_ax_left = grid_points_3d[any_index, -1, 0]
                coordinate_ax_right = grid_points_3d[any_index, 0, 0]
                grid_values_to_imshow = np.flip(grid_values_to_imshow, axis=1).copy()

            if grid_points_3d[-1, any_index, 1] < grid_points_3d[0, any_index, 1]:
                coordinate_ax_bottom = grid_points_3d[-1, any_index, 1]
                coordinate_ax_top = grid_points_3d[0, any_index, 1]
            else:
                coordinate_ax_bottom = grid_points_3d[0, any_index, 1]
                coordinate_ax_top = grid_points_3d[-1, any_index, 1]
                grid_values_to_imshow = np.flip(grid_values_to_imshow, axis=0).copy()
            self.ax_latent_space.imshow(grid_values_to_imshow,
                                        extent=[coordinate_ax_left,
                                                coordinate_ax_right,
                                                coordinate_ax_bottom,
                                                coordinate_ax_top],
                                        vmin=vmin,
                                        vmax=vmax,
                                        **self.params_imshow)
            ctr = self.ax_latent_space.contour(grid_points_3d[:, :, 0],
                                               grid_points_3d[:, :, 1],
                                               grid_values_to_contour, 6, colors='k')
            plt.setp(ctr.collections, path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                                                    path_effects.Normal()])
            clbls = self.ax_latent_space.clabel(ctr)
            plt.setp(clbls, path_effects=[path_effects.Stroke(linewidth=1, foreground='white'),
                                          path_effects.Normal()])

        # Plot latent variables
        if self.multiple_marker is None:
            self.ax_latent_space.scatter(self.Z[self.mask_latent_variables, 0],
                                         self.Z[self.mask_latent_variables, 1],
                                         **self.params_scatter)
        else:
            unique_markers = np.unique(self.multiple_marker)
            for marker in unique_markers:
                mask = (self.multiple_marker == marker)
                mask = self.mask_latent_variables & mask
                self.ax_latent_space.scatter(
                    self.Z[mask, 0],
                    self.Z[mask, 1],
                    marker=marker,
                    **self.params_scatter
                )
        # Write label
        if self.label_data is None:
            pass
        else:
            if self.is_show_all_label_data:
                for z, noise, label in zip(self.Z[self.mask_latent_variables],
                                           self.noise_label[self.mask_latent_variables],
                                           self.label_data[self.mask_latent_variables]):
                    point_label = z + noise
                    text = self.ax_latent_space.text(point_label[0], point_label[1], label,
                                                     ha='center', va='bottom', color='black')
                    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                           path_effects.Normal()])
            else:
                if self.index_data_label_shown is not None:
                    if self.mask_latent_variables[self.index_data_label_shown]:
                        text = self.ax_latent_space.text(self.Z[self.index_data_label_shown, 0],
                                                         self.Z[self.index_data_label_shown, 1],
                                                         self.label_data[self.index_data_label_shown],
                                                         ha='center', va='bottom', color='black')
                        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                               path_effects.Normal()]
                                              )
                else:
                    pass

        # Draw clicked coordinates
        if self.click_point_latent_space is None:
            pass
        else:
            coordinate = self.grid_points[self.click_point_latent_space]
            self.ax_latent_space.plot(coordinate[0], coordinate[1],
                                      ".", color="red", ms=20, fillstyle="none")

        self.fig.show()

    def _draw_feature_bars(self):
        self.ax_feature_bars.cla()
        self.feature_bars = self.ax_feature_bars.bar(self.label_feature, self.clicked_mapping)
        if self.selected_feature is not None:
            self.feature_bars[self.selected_feature].set_color('r')
        self.ax_feature_bars.set_ylim(self.X.min(), self.X.max() * 1.05)
        if self.click_point_latent_space is None:
            title = self.title_feature_bars + ' (Dataset mean)'
        else:
            title = self.title_feature_bars
        self.ax_feature_bars.set_title(title)
        self.ax_feature_bars.set_xticklabels(labels=self.label_feature, fontsize=8, rotation=270)
        self.fig.show()

    def __calc_nearest_grid_point(self, click_coordinates):
        distance = dist.cdist(self.grid_points, click_coordinates[None, :])
        index_nearest = np.argmin(distance.ravel())
        return index_nearest

    def __calc_nearest_latent_variable(self, click_coordinates, retdist=False):
        distance = dist.cdist(self.Z, click_coordinates[None, :], metric='euclidean')
        index_nearest = np.argmin(distance.ravel())
        dist_min = distance.min()
        if retdist:
            return index_nearest, dist_min
        else:
            return index_nearest

    def __unflatten_grid_array(self, grid_array):
        if grid_array.shape[0] == np.prod(self.n_grid_points):
            return np.squeeze(grid_array.reshape(np.append(self.n_grid_points, -1)))
        else:
            raise ValueError('arg shape {} is not consistent'.format(grid_array.shape))
