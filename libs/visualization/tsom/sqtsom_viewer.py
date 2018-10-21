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

class TSOM2_Umatrix:
    def __init__(self, z1=None,z2=None, x=None, sigma1=0.2,sigma2=0.2, resolution=100, labels1=None, labels2=None, fig_size=[15,15], cmap_type='jet'):
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
        self.Fig2 = plt.figure(figsize=(fig_size[0], fig_size[1]))
        self.Map1 = self.Fig1.add_subplot(1, 1, 1)
        self.Map2 = self.Fig2.add_subplot(1, 1, 1)
        self.Cmap_type = cmap_type
        self.labels1 = labels1
        self.labels2 = labels2

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

        # U-matrix表示
        self.Map1.set_title("U-matrix1")
        self.Map1.imshow(U_matrix_val, interpolation='spline36',
                      extent=[self.Zeta1[:, 0].min(), self.Zeta1[:, 0].max(),
                              self.Zeta1[:, 1].max(), self.Zeta1[:, 1].min()],
                   cmap=self.Cmap_type, vmax=0.5, vmin=-0.5)

        self.Map2.set_title("U-matrix2")
        self.Map2.imshow(U_matrix_val2, interpolation='spline36',
                         extent=[self.Zeta2[:, 0].min(), self.Zeta2[:, 0].max(),
                                 self.Zeta2[:, 1].max(), self.Zeta2[:, 1].min()],
                         cmap=self.Cmap_type, vmax=0.5, vmin=-0.5)

        # ラベルの表示
        self.Map1.scatter(x=self.Z1[:, 0], y=self.Z1[:, 1], c='k')
        self.Map2.scatter(x=self.Z2[:, 0], y=self.Z2[:, 1], c='k')
        umatrix_val = (np.random.rand(20, 20) - 0.5) * 2

        if self.labels1 is None:
            self.labels1 = np.arange(self.N) + 1

        if self.labels2 is None:
            self.labels2 = np.arange(self.N2) + 1

        # 勝者位置が重なった時用の処理
        epsilon = 0.04 * (self.Z1.max() - self.Z1.min())
        for i in range(self.Z1.shape[0]):
            count = 0
            for j in range(i):
                if np.allclose(self.Z1[j, :], self.Z1[i, :]):
                    count += 1
            self.Map1.text(self.Z1[i, 0], self.Z1[i, 1] + epsilon * count, self.labels1[i], color='k')

        epsilon2 = 0.04 * (self.Z2.max() - self.Z2.min())
        for i in range(self.Z2.shape[0]):
            count = 0
            for j in range(i):
                if np.allclose(self.Z2[j, :], self.Z2[i, :]):
                    count += 1
            self.Map2.text(self.Z2[i, 0], self.Z2[i, 1] + epsilon2 * count, self.labels2[i], color='k')
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

class TSOM2_ComponentPlane:
    def __init__(self, x, y, winner1, winner2, fig_size=None, label1=None, label2=None):
        # ---------- 参照テンソルとデータ ---------- #
        self.Mode1_Num = y.shape[0]
        self.Mode2_Num = y.shape[1]
        if y.ndim == 2:
            # 1次元の場合
            self.Dim = 1
            self.Y = y[:, :, np.newaxis]
            self.X = x[:, :, np.newaxis]
        else:
            # 2次元の場合
            self.Dim = y.shape[2]
            self.Y = y
            self.X = x

        # ---------- 勝者 ---------- #
        self.Winner1 = winner1
        self.Winner2 = winner2

        # ----------コンポーネントプレーン用---------- #
        self.Map1_click_unit = 0  # Map0のクリック位置
        self.Map2_click_unit = 0  # Map1のクリック位置
        self.map1x_num = int(np.sqrt(self.Mode1_Num))  # マップの1辺を算出（正方形が前提）
        self.map2x_num = int(np.sqrt(self.Mode2_Num))  # マップの1辺を算出（正方形が前提）

        # マップ上の座標
        map1x = np.arange(self.map1x_num)
        map1y = -np.arange(self.map1x_num)
        map1x_pos, map1y_pos = np.meshgrid(map1x, map1y)
        self.Map1_position = np.c_[map1x_pos.ravel(), map1y_pos.ravel()]  # マップ上の座標
        map2x = np.arange(self.map2x_num)
        map2y = -np.arange(self.map2x_num)
        map2x_pos, map2y_pos = np.meshgrid(map2x, map2y)
        self.Map2_position = np.c_[map2x_pos.ravel(), map2y_pos.ravel()]  # マップ上の座標

        # コンポーネントプレーン
        self.__calc_component(1)
        self.__calc_component(2)
        self.click_map = 0

        # ----------描画用---------- #
        self.Mapsize = np.sqrt(y.shape[0])
        if fig_size is None:
            self.Fig = plt.figure(figsize=(12, 6))
        else:
            self.Fig = plt.figure(figsize=fig_size)
        self.Map1 = self.Fig.add_subplot(1, 2, 1)
        self.Map2 = self.Fig.add_subplot(1, 2, 2)
        # self.Fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

        # 枠線と目盛りの消去
        self.Map1.spines["right"].set_color("none")
        self.Map1.spines["left"].set_color("none")
        self.Map1.spines["top"].set_color("none")
        self.Map1.spines["bottom"].set_color("none")
        self.Map2.spines["right"].set_color("none")
        self.Map2.spines["left"].set_color("none")
        self.Map2.spines["top"].set_color("none")
        self.Map2.spines["bottom"].set_color("none")
        self.Map1.tick_params(labelbottom='off', color='white')
        self.Map1.tick_params(labelleft='off')
        self.Map2.tick_params(labelbottom='off', color='white')
        self.Map2.tick_params(labelleft='off')

        # textboxのプロパティ
        self.bbox_labels = dict(fc="gray", ec="black", lw=2, alpha=0.5)
        self.bbox_mouse = dict(fc="yellow", ec="black", lw=2, alpha=0.9)

        # 勝者が被った場合にラベルが重ならないようにするためのノイズ
        self.noise_map1 = (np.random.rand(self.Winner1.shape[0], 2) - 0.5)
        self.noise_map2 = (np.random.rand(self.Winner2.shape[0], 2) - 0.5)

    # ------------------------------ #
    # --- イベント時の処理 ----------- #
    # ------------------------------ #
    # クリック時の処理
    def __onclick_fig(self, event):
        if event.xdata is not None:
            # クリック位置取得
            click_pos = np.random.rand(1, 2)
            click_pos[0, 0] = event.xdata
            click_pos[0, 1] = event.ydata

            if event.inaxes == self.Map1.axes:
                # 左のマップをクリックした時
                self.Map1_click_unit = self.__calc_arg_min_unit(self.Map1_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(2)
                self.click_map = 1

            elif event.inaxes == self.Map2.axes:
                # 右のマップをクリックした時
                self.Map2_click_unit = self.__calc_arg_min_unit(self.Map2_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(1)
                self.click_map = 2

            else:
                return

            # コンポーネントプレーン表示
            self.__draw_map1()
            self.__draw_map2()
            self.__draw_click_point()

    # マウスオーバー時(in)の処理
    def __mouse_over_fig(self, event):
        if event.xdata is not None:
            # マウスカーソル位置取得
            click_pos = np.random.rand(1, 2)
            click_pos[0, 0] = event.xdata
            click_pos[0, 1] = event.ydata

            if event.inaxes == self.Map1.axes:
                # 左マップのマウスオーバー処理
                mouse_over_unit = self.__calc_arg_min_unit(self.Map1_position, click_pos)
                self.__draw_mouse_over_label_map1(mouse_over_unit)

            elif event.inaxes == self.Map2.axes:
                # 右のマップのマウスオーバー処理
                mouse_over_unit = self.__calc_arg_min_unit(self.Map2_position, click_pos)
                self.__draw_mouse_over_label_map2(mouse_over_unit)

            self.__draw_click_point()
            self.Fig.show()

    # マウスオーバー時(out)の処理
    def __mouse_leave_fig(self, event):
        self.__draw_map1()
        self.__draw_map2()
        self.__draw_click_point()

    # ------------------------------ #
    # --- 描画 ---------------------- #
    # ------------------------------ #
    def draw_map_conponentplane(self):
        # コンポーネントの初期表示(左下が0番目のユニットが来るように行列を上下反転している)
        self.__draw_map1()
        self.__draw_map2()
        self.__draw_click_point()

        # クリックイベント
        self.Fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)

        # マウスオーバーイベント
        self.Fig.canvas.mpl_connect('motion_notify_event', self.__mouse_over_fig)
        self.Fig.canvas.mpl_connect('axes_leave_event', self.__mouse_leave_fig)
        plt.show()

    # ------------------------------ #
    # --- ラベルの描画 --------------- #
    # ------------------------------ #
    # 左のマップ
    def __draw_label_map1(self):
        epsilon = 0.02 * (self.Map1_position.max() - self.Map1_position.min())
        for i in range(self.Winner1.shape[0]):
            self.Map1.text(self.Map1_position[self.Winner1[i], 0] + epsilon * self.noise_map1[i, 0],
                           self.Map1_position[self.Winner1[i], 1] + epsilon * self.noise_map1[i, 1],
                           i + 1, ha='center', va='bottom', color='black')

            self.Map1.plot(self.Map1_position[self.Winner1[i], 0] + epsilon * self.noise_map1[i, 0],
                           self.Map1_position[self.Winner1[i], 1] + epsilon * self.noise_map1[i, 1],
                           marker='.', color="white", ms=15, markeredgecolor='black', markeredgewidth=1)
        self.Fig.show()

    # 右のマップ
    def __draw_label_map2(self):
        epsilon = 0.02 * (self.Map2_position.max() - self.Map2_position.min())
        for i in range(self.Winner2.shape[0]):
            self.Map2.text(self.Map2_position[self.Winner2[i], 0] + epsilon * self.noise_map2[i, 0],
                           self.Map2_position[self.Winner2[i], 1] + epsilon * self.noise_map2[i, 1],
                           i + 1, ha='center', va='bottom', color='black')

            self.Map2.plot(self.Map2_position[self.Winner2[i], 0] + epsilon * self.noise_map2[i, 0],
                           self.Map2_position[self.Winner2[i], 1] + epsilon * self.noise_map2[i, 1],
                           marker='.', color="white", ms=15, markeredgecolor='black', markeredgewidth=1)
        self.Fig.show()

    # ------------------------------ #
    # --- ラベルの描画(マウスオーバ時) - #
    # ------------------------------ #
    # 左のマップ
    def __draw_mouse_over_label_map1(self, mouse_over_unit):
        wine_labels = " "
        # for i in range(self.Winner0.shape[0]):
        #     if mouse_over_unit == self.Winner0[i]:
        #         if len(wine_labels) <= 1:
        #             wine_labels = self.labels0[i]
        #             temp = i
        #         else:
        #             wine_labels = wine_labels + "\n" + self.labels0[i]
        # if len(wine_labels) > 1:
        #     if self.radio_click_flg == 0:
        #         self.__draw_map0()
        #     elif self.radio_click_flg == 1:
        #         self.__draw_radio1_data()
        #         self.__draw_map0()
        #     elif self.radio_click_flg == 2:
        #         self.__draw_radio2_data()
        #         self.__draw_map0()
        #     if self.Winner0[temp] % self.map0x_num < self.map0x_num / 2.0:
        #         self.Map0.text(self.Map0_position[mouse_over_unit, 0], self.Map0_position[mouse_over_unit, 1],
        #                        wine_labels, color='black', ha='left', va='center', bbox=self.bbox_mouse)
        #     else:
        #         self.Map0.text(self.Map0_position[mouse_over_unit, 0], self.Map0_position[mouse_over_unit, 1],
        #                        wine_labels, color='black', ha='right', va='center', bbox=self.bbox_mouse)

    # 右のマップ
    def __draw_mouse_over_label_map2(self, mouse_over_unit):
        chemical_labels = " "
        # for i in range(self.Winner2.shape[0]):
        #     if mouse_over_unit == self.Winner2[i]:
        #         if len(chemical_labels) <= 1:
        #             chemical_labels = self.labels2[i]
        #             temp = i
        #         else:
        #             chemical_labels = chemical_labels + "\n" + self.labels2[i]
        # if len(chemical_labels) > 1:
        #     self.__draw_map2()
        #     if self.Winner2[temp] % self.map2x_num < self.map2x_num / 2.0:
        #         self.Map2.text(self.Map2_position[mouse_over_unit, 0], self.Map2_position[mouse_over_unit, 1],
        #                        chemical_labels, color='black', ha='left', va='center', bbox=self.bbox_mouse)
        #     else:
        #         self.Map2.text(self.Map2_position[mouse_over_unit, 0], self.Map2_position[mouse_over_unit, 1],
        #                        chemical_labels, color='black', ha='right', va='center', bbox=self.bbox_mouse)

    # ------------------------------ #
    # --- クリック位置の描画 ---------- #
    # ------------------------------ #
    def __draw_click_point(self):
        if self.click_map == 1:
            self.Map1.plot(self.Map1_position[self.Map1_click_unit, 0], self.Map1_position[self.Map1_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
        elif self.click_map == 2:
            self.Map2.plot(self.Map2_position[self.Map2_click_unit, 0], self.Map2_position[self.Map2_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
        self.Fig.show()

    # ------------------------------ #
    # --- コンポーネントプレーン表示 --- #
    # ------------------------------ #
    def __draw_map1(self):
        self.Map1.cla()
        # self.Map1.set_xlabel("Wine Map")
        self.__draw_label_map1()
        self.Map1.imshow(self.Map1_val[::], interpolation='spline36',
                         extent=[0, self.Map1_val.shape[0] - 1, -self.Map1_val.shape[1] + 1, 0], cmap="rainbow")
        self.Map1.set_xlim(-1, self.Mapsize)
        self.Map1.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    def __draw_map2(self):
        self.Map2.cla()
        # self.Map2.set_xlabel("Aroma Map")
        self.Map2.xaxis.set_label_coords(0.5, -0.1)
        self.__draw_label_map2()
        self.Map2.imshow(self.Map2_val[::], interpolation='spline36',
                         extent=[0, self.Map2_val.shape[0] - 1, -self.Map2_val.shape[1] + 1, 0], cmap="rainbow")
        self.Map2.set_xlim(-1, self.Mapsize)
        self.Map2.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    # ------------------------------ #
    # --- コンポーネント値の算出 ------ #
    # ------------------------------ #
    def __calc_component(self, map_num):
        if map_num == 1:
            temp1 = self.Y[:, self.Map2_click_unit, :]
            self.Map1_val = np.sqrt(np.sum(temp1 * temp1, axis=1)).reshape([self.map1x_num, self.map1x_num])
        else:
            temp2 = self.Y[self.Map1_click_unit, :, :]
            self.Map2_val = np.sqrt(np.sum(temp2 * temp2, axis=1)).reshape([self.map2x_num, self.map2x_num])

    # ------------------------------ #
    # --- 最近傍ユニット算出 ---------- #
    # ------------------------------ #
    @staticmethod
    def __calc_arg_min_unit(zeta, click_point):
        distance = dist.cdist(zeta, click_point)
        unit = np.argmin(distance, axis=0)
        return unit[0]

class ItemProperties(object):
    def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
                 alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha

        self.labelcolor_rgb = colors.to_rgba(labelcolor)[:3]
        self.bgcolor_rgb = colors.to_rgba(bgcolor)[:3]


class MenuItem(artist.Artist):
    parser = mathtext.MathTextParser("Bitmap")
    padx = 5
    pady = 5

    def __init__(self, fig, labelstr, props=None, hoverprops=None,
                 on_select=None):
        artist.Artist.__init__(self)

        self.set_figure(fig)
        self.labelstr = labelstr

        if props is None:
            props = ItemProperties()

        if hoverprops is None:
            hoverprops = ItemProperties()

        self.props = props
        self.hoverprops = hoverprops

        self.on_select = on_select

        x, self.depth = self.parser.to_mask(
            labelstr, fontsize=props.fontsize, dpi=fig.dpi)

        if props.fontsize != hoverprops.fontsize:
            raise NotImplementedError(
                'support for different font sizes not implemented')

        self.labelwidth = x.shape[1]
        self.labelheight = x.shape[0]

        self.labelArray = np.zeros((x.shape[0], x.shape[1], 4))
        self.labelArray[:, :, -1] = x/255.

        self.label = image.FigureImage(fig, origin='upper')
        self.label.set_array(self.labelArray)

        # we'll update these later
        self.rect = patches.Rectangle((0, 0), 1, 1)

        self.set_hover_props(False)

        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        over, junk = self.rect.contains(event)
        if not over:
            return

        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h):
        print(x, y, w, h)
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)

        self.label.ox = x + self.padx
        self.label.oy = y - self.depth + self.pady/2.

        self.rect._update_patch_transform()
        self.hover = False

    def draw(self, renderer):
        self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        if b:
            props = self.hoverprops
        else:
            props = self.props

        r, g, b = props.labelcolor_rgb
        self.labelArray[:, :, 0] = r
        self.labelArray[:, :, 1] = g
        self.labelArray[:, :, 2] = b
        self.label.set_array(self.labelArray)
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        'check the hover status of event and return true if status is changed'
        b, junk = self.rect.contains(event)

        changed = (b != self.hover)

        if changed:
            self.set_hover_props(b)

        self.hover = b
        return changed


class Menu(object):
    def __init__(self, fig, menuitems):
        self.figure = fig
        fig.suppressComposite = True

        self.menuitems = menuitems
        self.numitems = len(menuitems)

        maxw = max([item.labelwidth for item in menuitems])
        maxh = max([item.labelheight for item in menuitems])

        totalh = self.numitems*maxh + (self.numitems + 1)*2*MenuItem.pady

        x0 = 100
        y0 = 400

        width = maxw + 2*MenuItem.padx
        height = maxh + MenuItem.pady

        for item in menuitems:
            left = x0
            bottom = y0 - maxh - MenuItem.pady

            item.set_extent(left, bottom, width, height)

            fig.artists.append(item)
            y0 -= maxh + MenuItem.pady

        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        draw = False
        for item in self.menuitems:
            draw = item.set_hover(event)
            if draw:
                self.figure.canvas.draw()
                break
