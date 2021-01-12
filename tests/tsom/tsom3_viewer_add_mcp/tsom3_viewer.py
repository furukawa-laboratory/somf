import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from matplotlib.widgets import RadioButtons

#color bar
from matplotlib.colors import Normalize
import mpl_toolkits.axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(2)

class TSOM3_Viewer:
    def __init__(self, y, winner1, winner2, winner3, fig_size=None, label1=None, label2=None, label3=None, button_label=None, view1_title=None, view2_title=None, view3_title=None):
        # ---------- 参照テンソルとデータ ---------- #
        self.Mode1_Num = y.shape[0]
        self.Mode2_Num = y.shape[1]
        self.Mode3_Num = y.shape[2]
        if y.ndim == 3:
            # 3次元の場合
            self.Dim = y.shape[2]
            self.Y = y[:, :, :, np.newaxis]
        else:
            # 4次元以降
            self.Dim = y.shape[3]
            self.Y = y

        # ---------- 勝者 ---------- #
        self.Winner1 = winner1
        self.Winner2 = winner2
        self.Winner3 = winner3

        # ----------コンポーネントプレーン用---------- #
        self.Map1_click_unit = 0  # Map1のクリック位置
        self.Map2_click_unit = 0  # Map2のクリック位置
        self.Map3_click_unit = 0  # Map3のクリック位置
        self.Radio_click_unit = 0  # add machida Radioのクリック位置
        self.Radio_click_unit2 = 0  # add select abs or rel
        self.map1x_num = int(np.sqrt(self.Mode1_Num))  # マップの1辺を算出（正方形が前提）
        self.map2x_num = int(np.sqrt(self.Mode2_Num))  # マップの1辺を算出（正方形が前提）
        self.map3x_num = int(np.sqrt(self.Mode3_Num))  # マップの1辺を算出（正方形が前提）

        # -----------マップ管理用フラグ-------------
        self.map1_select_flag = 1
        self.map2_select_flag = 1
        self.map3_select_flag = 1

        # マップ上の座標
        map1x = np.arange(self.map1x_num)
        map1y = -np.arange(self.map1x_num)
        map1x_pos, map1y_pos = np.meshgrid(map1x, map1y)
        self.Map1_position = np.c_[map1x_pos.ravel(), map1y_pos.ravel()]  # マップ上の座標
        map2x = np.arange(self.map2x_num)
        map2y = -np.arange(self.map2x_num)
        map2x_pos, map2y_pos = np.meshgrid(map2x, map2y)
        self.Map2_position = np.c_[map2x_pos.ravel(), map2y_pos.ravel()]  # マップ上の座標
        map3x = np.arange(self.map3x_num)
        map3y = -np.arange(self.map3x_num)
        map3x_pos, map3y_pos = np.meshgrid(map3x, map3y)
        self.Map3_position = np.c_[map3x_pos.ravel(), map3y_pos.ravel()]  # マップ上の座標

        # label
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3
        self.button_label = button_label

        # title
        self.view1_title = view1_title
        if self.view1_title is None:
            self.view1_title = 'View 1'
        self.view2_title = view2_title
        if self.view2_title is None:
            self.view2_title = 'View 2'
        self.view3_title = view3_title
        if self.view3_title is None:
            self.view3_title = 'View 3'


        if button_label is None:#radioボタンにラベルが与えられなかったときの処理(観測データ次元分の番号を振る)
            self.button_label=np.arange(self.Dim)
            values = np.arange(self.Dim)
            #radioボタンにラベルをはる際に辞書を作成
            dict_keys = []
            for i in np.arange(self.Dim):
                dict_keys.append(str(self.button_label[i]))
            self.hzdict = dict(zip(dict_keys, values))  # e.g.Deskwork_or_studyingが与えられたら0を返す
        else:
            values = np.arange(self.Dim)
            # radioボタンにラベルをはる際に辞書を作成
            dict_keys = []
            for i in np.arange(self.Dim):
                dict_keys.append(str(self.button_label[i]))
            self.hzdict = dict(zip(dict_keys, values))  # e.g.Deskwork_or_studyingが与えられたら0を返す

        # コンポーネントプレーン
        self.__calc_component(1)
        self.__calc_component(2)
        self.__calc_component(3)
        self.click_map = 0

        # ----------描画用---------- #
        self.Mapsize = np.sqrt(y.shape[0])
        if fig_size is None:
            self.Fig = plt.figure(figsize=(15, 6))
        else:
            self.Fig = plt.figure(figsize=fig_size)
        plt.subplots_adjust(left=0)#added property
        plt.subplots_adjust(right=0.85)#0.7
        plt.subplots_adjust(wspace=0)
        self.Map1 = self.Fig.add_subplot(1, 3, 1)
        self.Map1.set_title(self.view1_title)
        self.Map2 = self.Fig.add_subplot(1, 3, 2)
        self.Map2.set_title(self.view2_title)
        self.Map3 = self.Fig.add_subplot(1, 3, 3)
        self.Map3.set_title(self.view3_title)
        rax = plt.axes([0.92, 0.25, 0.1, 0.5], facecolor='lightgoldenrodyellow',aspect='equal')
        if not button_label is None:
            self.radio = RadioButtons(rax, button_label)
        else:
            self.radio = RadioButtons(rax, np.arange(self.Dim))
        self.count_click = None

        # 絶対と相対の切り替え用
        rax2 = plt.axes([0.92, 0.50, 0.1, 0.75], facecolor='lightgoldenrodyellow', aspect='equal')
        self.radio2 = RadioButtons(rax2, ["relatively", "absolutely"])
        self.count_click2 = None

        # 枠線と目盛りの消去
        self.Map1.spines["right"].set_color("none")
        self.Map1.spines["left"].set_color("none")
        self.Map1.spines["top"].set_color("none")
        self.Map1.spines["bottom"].set_color("none")

        self.Map2.spines["right"].set_color("none")
        self.Map2.spines["left"].set_color("none")
        self.Map2.spines["top"].set_color("none")
        self.Map2.spines["bottom"].set_color("none")

        self.Map3.spines["right"].set_color("none")
        self.Map3.spines["left"].set_color("none")
        self.Map3.spines["top"].set_color("none")
        self.Map3.spines["bottom"].set_color("none")

        self.Map1.tick_params(labelbottom='off', color='white')
        self.Map1.tick_params(labelleft='off')

        self.Map2.tick_params(labelbottom='off', color='white')
        self.Map2.tick_params(labelleft='off')

        self.Map3.tick_params(labelbottom='off', color='white')
        self.Map3.tick_params(labelleft='off')

        # textboxのプロパティ
        self.bbox_labels = dict(fc="gray", ec="black", lw=2, alpha=0.5)
        self.bbox_mouse = dict(fc="yellow", ec="black", lw=2, alpha=0.9)

        # 勝者が被った場合にラベルが重ならないようにするためのノイズ
        self.noise_map1 = (np.random.rand(self.Winner1.shape[0], 2) - 0.5)
        self.noise_map2 = (np.random.rand(self.Winner2.shape[0], 2) - 0.5)
        self.noise_map3 = (np.random.rand(self.Winner3.shape[0], 2) - 0.5)

    def hzfunc(self, label): #radioボタンを押した時の処理
        if self.count_click == self.hzdict[label]:
            return
        else:
            self.count_click = self.hzdict[label]
            self.Radio_click_unit = self.hzdict[label]
            self.__calc_component(1)
            self.__calc_component(2)
            self.__calc_component(3)
            self.__draw_map1()
            self.__draw_map2()
            self.__draw_map3()
            self.__draw_click_point()

    def colorfunc(self, label):
        colordict = {'relatively': 0, 'absolutely': 1}
        self.Radio_click_unit2 = colordict[label]

    # ------------------------------ #
    # --- イベント時の処理 ----------- #
    # ------------------------------ #
    # クリック時の処理
    def __onclick_fig(self, event):
        #左クリックされたとき
        if event.button == 1:
            # クリック位置取得
            click_pos = np.random.rand(1, 2)
            click_pos[0, 0] = event.xdata
            click_pos[0, 1] = event.ydata

            if event.inaxes == self.Map1.axes:
                self.map1_select_flag = 1
                # マップ１をクリック
                self.Map1_click_unit = self.__calc_arg_min_unit(self.Map1_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(2)
                self.__calc_component(3)
                self.click_map = 1

            elif event.inaxes == self.Map2.axes:
                self.map2_select_flag = 1
                # マップ2をクリックした時
                self.Map2_click_unit = self.__calc_arg_min_unit(self.Map2_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(1)
                self.__calc_component(3)
                self.click_map = 2

            elif event.inaxes == self.Map3.axes:
                self.map3_select_flag = 1
                # マップ３をクリックした時
                self.Map3_click_unit = self.__calc_arg_min_unit(self.Map3_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(1)
                self.__calc_component(2)
                self.click_map = 3

            else:
                return
            # コンポーネントプレーン表示
            self.__draw_map1()
            self.__draw_map2()
            self.__draw_map3()
            self.__draw_click_point()

        #右クリックされたとき
        if event.button == 3:
            # クリック位置取得
            click_pos = np.random.rand(1, 2)
            click_pos[0, 0] = event.xdata
            click_pos[0, 1] = event.ydata

            if event.inaxes == self.Map1.axes:
                self.map1_select_flag = 0
                # マップ１をクリック
                self.Map1_click_unit = self.__calc_arg_min_unit(self.Map1_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(2)
                self.__calc_component(3)
                self.click_map = 1

            elif event.inaxes == self.Map2.axes:
                self.map2_select_flag = 0
                # マップ2をクリックした時
                self.Map2_click_unit = self.__calc_arg_min_unit(self.Map2_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(1)
                self.__calc_component(3)
                self.click_map = 2

            elif event.inaxes == self.Map3.axes:
                self.map3_select_flag = 0
                # マップ３をクリックした時
                self.Map3_click_unit = self.__calc_arg_min_unit(self.Map3_position, click_pos)
                # コンポーネント値計算
                self.__calc_component(1)
                self.__calc_component(2)
                self.click_map = 3

            else:
                return
            # コンポーネントプレーン表示
            self.__draw_map1()
            self.__draw_map2()
            self.__draw_map3()
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
        self.__draw_map3()
        # self.radio.on_clicked(self.hzfunc)
        self.__draw_click_point()

    # ------------------------------ #
    # --- 描画 ---------------------- #
    # ------------------------------ #


    def draw_map(self):
        # コンポーネントの初期表示(左下が0番目のユニットが来るように行列を上下反転している)
        self.__draw_map1()
        self.__draw_map2()
        self.__draw_map3()
        # self.radio.on_clicked(self.hzfunc)
        self.radio2.on_clicked(self.colorfunc)
        self.__draw_click_point()

        # divider = mpl_toolkits.axes_grid1.make_axes_locatable(self.Map3)
        # cax = divider.append_axes('right', '2%', pad=0.1)
        mappable0 = self.Map3.pcolormesh(self.Map3_val[::], cmap='bwr', norm=Normalize(vmin=np.min(self.Y), vmax=np.max(self.Y)))
        cax = plt.axes([0.88, 0.2, 0.010, 0.60]) # [左端からの距離、下からの距離、幅、高さ]
        self.cbar3 = self.Fig.colorbar(mappable0, cax=cax, ax=self.Map3, orientation="vertical")
        # self.cbar3 = self.Fig.colorbar(self.cmap3, cax=cax)
        self.cbar3.set_clim(np.min(self.Y), np.max(self.Y))

        # クリックイベント
        self.Fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)

        # マウスオーバーイベント
        self.Fig.canvas.mpl_connect('motion_notify_event', self.__mouse_over_fig)
        self.Fig.canvas.mpl_connect('axes_leave_event', self.__mouse_leave_fig)
        plt.show()

    # ------------------------------ #
    # --- ラベルの描画 --------------- #
    # ------------------------------ #
    # マップ１
    def __draw_label_map1(self):
        epsilon = 0.02 * (self.Map1_position.max() - self.Map1_position.min())
        if not self.label1 is None:#ラベルを与えばそのラベルを出力,そうでないなら出力しない
            for i in range(self.Winner1.shape[0]):
                self.Map1.text(self.Map1_position[self.Winner1[i], 0] + epsilon * self.noise_map1[i, 0],
                           self.Map1_position[self.Winner1[i], 1] + epsilon * self.noise_map1[i, 1],
                           self.label1[i], ha='center', va='bottom', color='black')
        self.Map1.scatter(self.Map1_position[self.Winner1[:], 0] + epsilon * self.noise_map1[:, 0],
                          self.Map1_position[self.Winner1[:], 1] + epsilon * self.noise_map1[:, 1],
                          c="white",linewidths=1,edgecolors="black")
        self.Fig.show()

    # マップ２
    def __draw_label_map2(self):
        epsilon = 0.02 * (self.Map2_position.max() - self.Map2_position.min())
        if not self.label2 is None:  # ラベルを与えばそのラベルを出力,そうでないなら出力しない
            for i in range(self.Winner2.shape[0]):
                self.Map2.text(self.Map2_position[self.Winner2[i], 0] + epsilon * self.noise_map2[i, 0],
                           self.Map2_position[self.Winner2[i], 1] + epsilon * self.noise_map2[i, 1],
                           self.label2[i], ha='center', va='bottom', color='black')
        self.Map2.scatter(self.Map2_position[self.Winner2[:], 0] + epsilon * self.noise_map2[:, 0],
                          self.Map2_position[self.Winner2[:], 1] + epsilon * self.noise_map2[:, 1],
                          c="white", linewidths=1, edgecolors="black")
        self.Fig.show()

    # マップ３
    def __draw_label_map3(self):
        epsilon = 0.02 * (self.Map3_position.max() - self.Map3_position.min())
        if not self.label3 is None:  # ラベルを与えばそのラベルを出力,そうでないなら出力しない
            for i in range(self.Winner3.shape[0]):
                self.Map3.text(self.Map3_position[self.Winner3[i], 0] + epsilon * self.noise_map3[i, 0],
                           self.Map3_position[self.Winner3[i], 1] + epsilon * self.noise_map3[i, 1],
                           self.label3[i], ha='center', va='bottom', color='black')
        self.Map3.scatter(self.Map3_position[self.Winner3[:], 0] + epsilon * self.noise_map3[:, 0],
                          self.Map3_position[self.Winner3[:], 1] + epsilon * self.noise_map3[:, 1],
                          c="white", linewidths=1, edgecolors="black")
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
        if self.map1_select_flag == 1 and self.map2_select_flag == 1 and self.map3_select_flag == 1:
            self.Map1.plot(self.Map1_position[self.Map1_click_unit, 0], self.Map1_position[self.Map1_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Map2.plot(self.Map2_position[self.Map2_click_unit, 0], self.Map2_position[self.Map2_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Map3.plot(self.Map3_position[self.Map3_click_unit, 0], self.Map3_position[self.Map3_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Fig.show()
        elif self.map1_select_flag == 0 and self.map2_select_flag == 1 and self.map3_select_flag == 1:
            self.Map2.plot(self.Map2_position[self.Map2_click_unit, 0], self.Map2_position[self.Map2_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Map3.plot(self.Map3_position[self.Map3_click_unit, 0], self.Map3_position[self.Map3_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Fig.show()
        elif self.map1_select_flag == 1 and self.map2_select_flag == 0 and self.map3_select_flag == 1:
            self.Map1.plot(self.Map1_position[self.Map1_click_unit, 0], self.Map1_position[self.Map1_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Map3.plot(self.Map3_position[self.Map3_click_unit, 0], self.Map3_position[self.Map3_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Fig.show()
        elif self.map1_select_flag == 1 and self.map2_select_flag == 1 and self.map3_select_flag == 0:
            self.Map1.plot(self.Map1_position[self.Map1_click_unit, 0], self.Map1_position[self.Map1_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Map2.plot(self.Map2_position[self.Map2_click_unit, 0], self.Map2_position[self.Map2_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Fig.show()
        elif self.map1_select_flag == 0 and self.map2_select_flag == 0 and self.map3_select_flag == 1:
            self.Map3.plot(self.Map3_position[self.Map3_click_unit, 0], self.Map3_position[self.Map3_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Fig.show()
        elif self.map1_select_flag == 0 and self.map2_select_flag == 1 and self.map3_select_flag == 0:
            self.Map2.plot(self.Map2_position[self.Map2_click_unit, 0], self.Map2_position[self.Map2_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Fig.show()
        elif self.map1_select_flag == 1 and self.map2_select_flag == 0 and self.map3_select_flag == 0:
            self.Map1.plot(self.Map1_position[self.Map1_click_unit, 0], self.Map1_position[self.Map1_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Fig.show()
        elif self.map1_select_flag == 0 and self.map2_select_flag == 0 and self.map3_select_flag == 0:
            self.Fig.show()

    # ------------------------------ #
    # --- コンポーネントプレーン表示 --- #
    # ------------------------------ #
    def __draw_map1(self):
        self.Map1.cla()
        self.Map1.set_title(self.view1_title)
        self.__draw_label_map1()
        if self.Radio_click_unit2 == 0:
            self.Map1.imshow(self.Map1_val[::], interpolation='spline36',
                         extent=[0, self.Map1_val.shape[0] - 1, -self.Map1_val.shape[1] + 1, 0], cmap="bwr")
        else:
            self.Map1.imshow(self.Map1_val[::], interpolation='spline36', vmin=np.min(self.Y), vmax=np.max(self.Y),
                             extent=[0, self.Map1_val.shape[0] - 1, -self.Map1_val.shape[1] + 1, 0], cmap="bwr")
        # print(self.Map1_val/np.max(self.Map1_val))
        self.Map1.set_xlim(-1, self.Mapsize)
        self.Map1.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    def __draw_map2(self):
        self.Map2.cla()
        self.Map2.set_title(self.view2_title)
        self.__draw_label_map2()
        if self.Radio_click_unit2 == 0:
            self.Map2.imshow(self.Map2_val[::], interpolation='spline36',
                         extent=[0, self.Map2_val.shape[0] - 1, -self.Map2_val.shape[1] + 1, 0], cmap="bwr")
        else:
            self.Map2.imshow(self.Map2_val[::], interpolation='spline36', vmin=np.min(self.Y), vmax=np.max(self.Y),
                             extent=[0, self.Map2_val.shape[0] - 1, -self.Map2_val.shape[1] + 1, 0], cmap="bwr")
        self.Map2.set_xlim(-1, self.Mapsize)
        self.Map2.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    def __draw_map3(self):
        self.Map3.cla()
        self.Map3.set_title(self.view3_title)
        self.__draw_label_map3()
        if self.Radio_click_unit2 == 0:
            self.Map3.imshow(self.Map3_val[::], interpolation='spline36',
                         extent=[0, self.Map3_val.shape[0] - 1, -self.Map3_val.shape[1] + 1, 0], cmap="bwr")
            # self.cmap3 = self.Map3.imshow(self.Map3_val[::], interpolation='spline36',
            #              extent=[0, self.Map3_val.shape[0] - 1, -self.Map3_val.shape[1] + 1, 0], cmap="bwr")
        else:
            self.Map3.imshow(self.Map3_val[::], interpolation='spline36', vmin=np.min(self.Y), vmax=np.max(self.Y),
                             extent=[0, self.Map3_val.shape[0] - 1, -self.Map3_val.shape[1] + 1, 0], cmap="bwr")
            # self.cmap3 = self.Map3.imshow(self.Map3_val[::], interpolation='spline36', vmin=np.min(self.Y), vmax=np.max(self.Y),
            #                  extent=[0, self.Map3_val.shape[0] - 1, -self.Map3_val.shape[1] + 1, 0], cmap="bwr")

        self.Map3.set_xlim(-1, self.Mapsize)
        self.Map3.set_ylim(-self.Mapsize, 1)
        self.Fig.show()



    # ------------------------------ #
    # --- コンポーネント値の算出 ------ #
    # ------------------------------ #
    def __calc_component(self, map_num):
        if map_num == 1:
            # conditional
            if self.map2_select_flag == 1 and self.map3_select_flag == 1:
                temp1 = self.Y[:, self.Map2_click_unit, self.Map3_click_unit, self.Radio_click_unit]
                self.Map1_val = temp1.reshape((self.map1x_num, self.map1x_num))
                #np.sqrt(np.sum(temp1 * temp1, axis=1)).reshape([self.map1x_num, self.map1x_num])
            # marginal
            elif self.map2_select_flag == 1 and self.map3_select_flag == 0:
                temp1 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=2)[:, self.Map2_click_unit]
                self.Map1_val = temp1.reshape((self.map1x_num, self.map1x_num))
                # np.sqrt(np.sum(temp1 * temp1, axis=1)).reshape([self.map1x_num, self.map1x_num])
            # marginal
            elif self.map2_select_flag == 0 and self.map3_select_flag == 1:
                temp1 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=1)[:, self.Map3_click_unit]
                self.Map1_val = temp1.reshape((self.map1x_num, self.map1x_num))
            # marginal
            elif self.map2_select_flag == 0 and self.map3_select_flag == 0:
                temp1 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=(1,2))
                self.Map1_val = temp1.reshape((self.map1x_num, self.map1x_num))
        if map_num == 2:
            # conditional
            if self.map1_select_flag == 1 and self.map3_select_flag == 1:
                temp2 = self.Y[self.Map1_click_unit, :, self.Map3_click_unit, self.Radio_click_unit]
                self.Map2_val = temp2.reshape((self.map2x_num, self.map2x_num))
                #np.sqrt(np.sum(temp2 * temp2, axis=1)).reshape([self.map2x_num, self.map2x_num])
            # marginal
            elif self.map1_select_flag == 1 and self.map3_select_flag == 0:
                temp2 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=2)[self.Map1_click_unit, :]
                self.Map2_val = temp2.reshape((self.map2x_num, self.map2x_num))
            # marginal
            elif self.map1_select_flag == 0 and self.map3_select_flag == 1:
                temp2 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=0)[:, self.Map3_click_unit]
                self.Map2_val = temp2.reshape((self.map2x_num, self.map2x_num))
            # marginal
            elif self.map1_select_flag == 0 and self.map3_select_flag == 0:
                temp2 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=(0, 2))
                self.Map2_val = temp2.reshape((self.map2x_num, self.map2x_num))
        else:
            # conditional
            if self.map1_select_flag == 1 and self.map2_select_flag == 1:
                temp3 = self.Y[self.Map1_click_unit, self.Map2_click_unit, :, self.Radio_click_unit]
                self.Map3_val = temp3.reshape((self.map3x_num, self.map3x_num))
                #np.sqrt(np.sum(temp2 * temp2, axis=1)).reshape([self.map2x_num, self.map2x_num])
            # marginal
            elif self.map1_select_flag == 1 and self.map2_select_flag == 0:
                temp3 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=1)[self.Map1_click_unit, :]
                self.Map3_val = temp3.reshape((self.map3x_num, self.map3x_num))
            # marginal
            elif self.map1_select_flag == 0 and self.map2_select_flag == 1:
                temp3 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=0)[self.Map2_click_unit, :]
                self.Map3_val = temp3.reshape((self.map3x_num, self.map3x_num))
            # marginal
            elif self.map1_select_flag == 0 and self.map2_select_flag == 0:
                temp3 = np.mean(self.Y[:, :, :, self.Radio_click_unit], axis=(0, 1))
                self.Map3_val = temp3.reshape((self.map3x_num, self.map3x_num))

        # if map_num == 1:
        #     temp1 = np.mean(self.Y[:, :, self.Radio_click_unit], axis=1)  # mode1のmarginal component planeの計算
        #     self.Map1_marginal_val = temp1.reshape((self.map1x_num,
        #                                             self.map1x_num))  # np.sqrt(np.sum(temp1 * temp1, axis=1)).reshape([self.map1x_num, self.map1x_num])
        # else:
        #     temp2 = np.mean(self.Y[:, :, self.Radio_click_unit], axis=0)  # mode2のmarginal component planeの計算
        #     self.Map2_marginal_val = temp2.reshape((self.map2x_num,
        #                                             self.map2x_num))  # np.sqrt

    # ------------------------------ #
    # --- 最近傍ユニット算出 ---------- #
    # ------------------------------ #
    @staticmethod
    def __calc_arg_min_unit(zeta, click_point):
        distance = dist.cdist(zeta, click_point)
        unit = np.argmin(distance, axis=0)
        return unit[0]
