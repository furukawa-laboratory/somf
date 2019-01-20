import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from matplotlib.widgets import RadioButtons
import pandas as pd

plt.rcParams['font.family'] = 'IPAexGothic'

# メモ
# y1    : (ユーザ)*(性格)　＋　(ユーザ)*(趣味)
# y2    : (ユーザ)*(乗り物)*(評価)
# mode0 : ユーザ(共通)
# mode1 : 性格
# mode1 : 趣味
# mode2 : 乗り物
# mode3 : 評価

np.random.seed(1)

class TSOMg:
    def __init__(self, y1, y2, y3, winner0, winner1, winner2, winner3, winner4, y_attribute):
        self.flg = 1

        # ----------参照ベクトル---------- #
        self.Mode0_Num = y1.shape[0]
        self.Mode1_Num = y1.shape[1]
        self.Mode2_Num = y2.shape[1]
        self.Mode3_Num = y3.shape[1]
        self.Mode4_Num = y3.shape[2]
        self.Dim = 1
        self.Y1 = y1[:, :, np.newaxis]
        self.Y2 = y2[:, :, np.newaxis]
        self.Y3 = y3[:, :, :, np.newaxis]
        self.Y_attribute = y_attribute
        self.Winner0 = winner0
        self.Winner1 = winner1
        self.Winner2 = winner2
        self.Winner3 = winner3
        self.Winner4 = winner4

        self.V_Min = []
        sigma = None
        msigma = None

        self.V_Min.append(msigma)
        self.V_Min.append(msigma)
        self.V_Min.append(msigma)
        self.V_Min.append(msigma)
        self.V_Min.append(msigma)
        self.V_Max = []
        self.V_Max.append(sigma)
        self.V_Max.append(sigma)
        self.V_Max.append(sigma)
        self.V_Max.append(sigma)
        self.V_Max.append(sigma)

        # ----------コンポーネントプレーン用---------- #
        self.Map0_click_unit = 15  # Map0のクリック位置(-1：クリックしていない)
        self.Map1_click_unit = -1  # Map1のクリック位置
        self.Map2_click_unit = -1  # Map2のクリック位置
        self.Map3_click_unit = 15  # Map3のクリック位置
        self.Map4_click_unit = 15  # Map4のクリック位置
        self.map0x_num = int(np.sqrt(self.Mode0_Num))  # マップの1辺を算出（正方形が前提）
        self.map1x_num = int(np.sqrt(self.Mode1_Num))  # マップの1辺を算出（正方形が前提）
        self.map2x_num = int(np.sqrt(self.Mode2_Num))  # マップの1辺を算出（正方形が前提）
        self.map3x_num = int(np.sqrt(self.Mode3_Num))  # マップの1辺を算出（正方形が前提）
        self.map4x_num = int(np.sqrt(self.Mode4_Num))  # マップの1辺を算出（正方形が前提）
        self.click_map = 0

        # マップ上の座標
        map0x = np.arange(self.map0x_num)
        map0y = -np.arange(self.map0x_num)
        map0x_pos, map0y_pos = np.meshgrid(map0x, map0y)
        self.Map0_position = np.c_[map0x_pos.ravel(), map0y_pos.ravel()]  # マップ上の座標
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
        map4x = np.arange(self.map4x_num)
        map4y = -np.arange(self.map4x_num)
        map4x_pos, map4y_pos = np.meshgrid(map4x, map4y)
        self.Map4_position = np.c_[map4x_pos.ravel(), map4y_pos.ravel()]  # マップ上の座標

        # コンポーネントプレーン
        self.__calc_propagation1(0)
        self.__calc_propagation0(1)
        self.__calc_propagation0(2)
        self.__calc_propagation0(3)
        self.__calc_propagation0(4)

        # ----------描画用---------- #
        self.Mapsize = np.sqrt(y1.shape[0])
        self.Fig = plt.figure(figsize=(16, 8))
        self.Map0 = self.Fig.add_subplot(242)
        self.Map1 = self.Fig.add_subplot(243)
        self.Map2 = self.Fig.add_subplot(247)
        self.Map3 = self.Fig.add_subplot(241)
        self.Map4 = self.Fig.add_subplot(245)

        self.Fig.subplots_adjust(left=0.01, bottom=0.01, right=1., top=1., wspace=-0.05, hspace=0.01)
        self.marker_size = [0 for i in range(self.Winner0.shape[0])]
        self.marker_type = [0 for i in range(self.Winner0.shape[0])]
        for i in range(self.Winner0.shape[0]):
            self.marker_size[i] = 15
            self.marker_type[i] = '.'
        self.moji_size = 8

        # 枠線と目盛りの消去
        if self.flg == 1:
            self.Map0.spines["right"].set_color("none")
            self.Map0.spines["left"].set_color("none")
            self.Map0.spines["top"].set_color("none")
            self.Map0.spines["bottom"].set_color("none")
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
            self.Map4.spines["right"].set_color("none")
            self.Map4.spines["left"].set_color("none")
            self.Map4.spines["top"].set_color("none")
            self.Map4.spines["bottom"].set_color("none")
        self.Map0.tick_params(labelbottom='off', color='white')
        self.Map0.tick_params(labelleft='off')
        self.Map1.tick_params(labelbottom='off', color='white')
        self.Map1.tick_params(labelleft='off')
        self.Map2.tick_params(labelbottom='off', color='white')
        self.Map2.tick_params(labelleft='off')
        self.Map3.tick_params(labelbottom='off', color='white')
        self.Map3.tick_params(labelleft='off')
        self.Map4.tick_params(labelbottom='off', color='white')
        self.Map4.tick_params(labelleft='off')

        # textboxのプロパティ
        self.bbox_labels = dict(fc="gray", ec="black", lw=2, alpha=0.5)
        self.bbox_mouse = dict(fc="yellow", ec="black", lw=2, alpha=0.9)

        # ラベル
        self.labels0 = ['']

        labels1 = pd.read_table("./data/personal_label_short.txt")
        self.labels1 = []
        self.labels2 = []
        for n in range(60):
            if n < 19:
                self.labels1.append(labels1.iat[n, 0])
            else:
                self.labels2.append(labels1.iat[n, 0])

        labels3 = pd.read_table("./data/car_label.txt")
        self.labels3 = []
        for n in range(10):
            self.labels3.append(labels3.iat[n, 0])

        labels4 = pd.read_table("./data/question_label_short.txt")
        self.labels4 = []
        for n in range(20):
            self.labels4.append(labels4.iat[n, 0])

        # ラジオボタン
        radio1_labels = ["男性", "女性"]
        self.radio1_dict = {"男性": 0, "女性": 1}
        radio2_labels = ["20代", "30代", "40代", "50代", "60代", "70代", "80代"]
        self.radio2_dict = {"20代": 0, "30代": 1, "40代": 2, "50代": 3, "60代": 4, "70代": 5, "80代": 6}
        radio3_labels = ["北海道", "東北", "関東", "中部", "近畿", "中国", "四国", "九州・沖縄県", "海外"]
        self.radio3_dict = {"北海道": 0, "東北": 1, "関東": 2, "中部": 3, "近畿": 4, "中国": 5, "四国": 6, "九州・沖縄県": 7,
                            "海外": 8}

        # rax1_x = 0.64
        # rax1_y = 0.88
        # rax1_h = 0.1
        # rax1_w = 0.05

        rax1_x = 0.748
        rax1_y = 0.778
        rax1_h = 0.2
        rax1_w = 0.1

        rax2_x = 0.748
        rax2_y = 0.55
        rax2_h = 0.2
        rax2_w = 0.1

        rax3_x = 0.86
        rax3_y = 0.77
        rax3_h = 0.2
        rax3_w = 0.12

        rax1 = plt.axes([rax1_x, rax1_y, rax1_w, rax1_h], facecolor='cyan', frameon=True, aspect='equal')
        rax2 = plt.axes([rax2_x, rax2_y, rax2_w, rax2_h], facecolor='cyan', frameon=True, aspect='equal')
        rax3 = plt.axes([rax3_x, rax3_y, rax3_w, rax3_h], facecolor='cyan', frameon=True, aspect='equal')
        self.radio1 = RadioButtons(rax1, radio1_labels)
        self.radio2 = RadioButtons(rax2, radio2_labels)
        self.radio3 = RadioButtons(rax3, radio3_labels)
        self.click_radio1_num = -1
        self.click_radio2_num = -1
        self.click_radio3_num = -1
        self.radio_click_flg = 0

        for circle in self.radio1.circles:
            circle.set_radius(0.08)
            circle.fill = None
            # circle.center = (self.radio.circles[0].center[0] + i*0.05, self.radio.circles[0].center[1] - i*0.04)
        for circle in self.radio2.circles:
            circle.set_radius(0.05)
            circle.fill = None
            # circle.center = (self.radio.circles[0].center[0] + i*0.05, self.radio.circles[0].center[1] - i*0.04)
        for circle in self.radio3.circles:
            circle.set_radius(0.04)
            # circle.height /= 5
            circle.fill = None
            # circle.center = (self.radio.circles[0].center[0] + i*0.05, self.radio.circles[0].center[1] - i*0.04)
        # rpos = rax3.get_position().get_points()
        # fh = self.Fig.get_figheight()
        # fw = self.Fig.get_figwidth()
        # rscale = (rpos[:, 1].ptp() / rpos[:, 0].ptp()) * (fh / fw)
        # for circ in self.radio3.circles:
        #     circ.height /= rscale*0.8

        # # ラジオボタンのテキストの位置
        # for label in self.radio1.labels:
        #     label.set_x(i*0.1)
        #     label.set_y(0.5)

        # 初期表示は表示しない
        self.radio1.circles[0].fill = None
        self.radio2.circles[0].fill = None
        self.radio3.circles[0].fill = None

        # ノイズ
        self.noise_map0 = (np.random.rand(self.Winner0.shape[0], 2) - 0.5)
        self.noise_map2 = (np.random.rand(self.Winner2.shape[0], 2) - 0.5)
        self.noise_map3 = (np.random.rand(self.Winner3.shape[0], 2) - 0.5)

    # ------------------------------ #
    # -----イベント時の処理----------- #
    # ------------------------------ #
    # クリック時の処理
    def __onclick_fig(self, event):
        if event.xdata is not None:
            # クリック位置取得
            click_pos = np.random.rand(1, 2)
            click_pos[0, 0] = event.xdata
            click_pos[0, 1] = event.ydata

            if event.inaxes == self.Map0.axes:
                # ユーザマップをクリックした時
                self.Map0_click_unit = self.__calc_arg_min_unit(self.Map0_position, click_pos)

                # コンポーネント値計算
                self.__calc_propagation0(1)
                self.__calc_propagation0(2)
                self.__calc_propagation0(3)
                self.__calc_propagation0(4)
                self.click_map = 0

            elif event.inaxes == self.Map1.axes:
                # 性格データマップをクリックした時
                self.Map1_click_unit = self.__calc_arg_min_unit(self.Map1_position, click_pos)

                # コンポーネント値計算
                self.__calc_propagation1(0)
                self.__calc_propagation1(2)
                self.__calc_propagation1(3)
                self.__calc_propagation1(4)
                self.click_map = 1

            elif event.inaxes == self.Map2.axes:
                # 趣味マップをクリックした時
                self.Map2_click_unit = self.__calc_arg_min_unit(self.Map2_position, click_pos)

                # コンポーネント値計算
                self.__calc_propagation2(0)
                self.__calc_propagation2(1)
                self.__calc_propagation2(3)
                self.__calc_propagation2(4)
                self.click_map = 2

            elif event.inaxes == self.Map3.axes:
                # 乗り物マップをクリックした時
                self.Map3_click_unit = self.__calc_arg_min_unit(self.Map3_position, click_pos)

                # コンポーネント値計算
                self.__calc_propagation3(0)
                self.__calc_propagation3(1)
                self.__calc_propagation3(2)
                self.__calc_propagation3(4)
                self.click_map = 3

            elif event.inaxes == self.Map4.axes:
                # 評価マップをクリックした時
                self.Map4_click_unit = self.__calc_arg_min_unit(self.Map4_position, click_pos)

                # コンポーネント値計算
                self.__calc_propagation4(0)
                self.__calc_propagation4(1)
                self.__calc_propagation4(2)
                self.__calc_propagation4(3)
                self.click_map = 4

            else:
                return

            # コンポーネントプレーン表示
            self.__draw_map0()
            self.__draw_map1()
            self.__draw_map2()
            self.__draw_map3()
            self.__draw_map4()
            self.__draw_click_point()

    # マウスオーバー時(in)の処理
    def __mouse_over_fig(self, event):
        if event.xdata is not None:
            # マウスカーソル位置取得
            click_pos = np.random.rand(1, 2)
            click_pos[0, 0] = event.xdata
            click_pos[0, 1] = event.ydata
            
            if event.inaxes == self.Map0.axes:
                # ユーザのマウスオーバー処理
                mouse_over_unit = self.__calc_arg_min_unit(self.Map0_position, click_pos)
                # self.__draw_mouse_over_label0(mouse_over_unit)

            elif event.inaxes == self.Map1.axes:
                # 性格データマップのマウスオーバー処理
                mouse_over_unit = self.__calc_arg_min_unit(self.Map1_position, click_pos)
                # self.__draw_mouse_over_label1(mouse_over_unit)

            elif event.inaxes == self.Map2.axes:
                # 趣味マップのマウスオーバー処理
                mouse_over_unit = self.__calc_arg_min_unit(self.Map2_position, click_pos)
                # self.__draw_mouse_over_label2(mouse_over_unit)

            elif event.inaxes == self.Map3.axes:
                # 乗り物マップのマウスオーバー処理
                mouse_over_unit = self.__calc_arg_min_unit(self.Map3_position, click_pos)
                # self.__draw_mouse_over_label3(mouse_over_unit)

            elif event.inaxes == self.Map4.axes:
                # 評価マップのマウスオーバー処理
                mouse_over_unit = self.__calc_arg_min_unit(self.Map4_position, click_pos)
                # self.__draw_mouse_over_label4(mouse_over_unit)

            self.__draw_click_point()
            self.Fig.show()

    # マウスオーバー時(out)の処理
    def __mouse_leave_fig(self, event):
        self.__draw_map0()
        self.__draw_map1()
        self.__draw_map2()
        self.__draw_map3()
        self.__draw_map4()
        self.__draw_click_point()

    # ラジオボタンをクリック
    def __radio1_click(self, label):
        if self.click_radio1_num == self.radio1_dict[label] and self.radio_click_flg == 1:
            self.click_radio1_num = -1
            self.radio_click_flg = 0

            for circle in self.radio1.circles:
                circle.fill = None

            self.__draw_map0()
            self.__draw_click_point()
        else:
            self.radio_click_flg = 1
            self.click_radio1_num = self.radio1_dict[label]
            for circle in self.radio1.circles:
                circle.fill = True
            for circle in self.radio2.circles:
                circle.fill = None
            for circle in self.radio3.circles:
                circle.fill = None
            self.click_radio2_num = -1
            self.click_radio3_num = -1

            self.__calc_radio_data()
            self.__draw_map0()
            self.__draw_click_point()

    def __radio2_click(self, label):
        if self.click_radio2_num == self.radio2_dict[label] and self.radio_click_flg == 1:
            self.click_radio2_num = -1
            self.radio_click_flg = 0

            for circle in self.radio2.circles:
                circle.fill = None

            self.__draw_map0()
            self.__draw_click_point()
        else:
            self.radio_click_flg = 1
            self.click_radio2_num = self.radio2_dict[label]
            for circle in self.radio2.circles:
                circle.fill = True
            for circle in self.radio1.circles:
                circle.fill = None
            for circle in self.radio3.circles:
                circle.fill = None
            self.click_radio1_num = -1
            self.click_radio3_num = -1

            self.__calc_radio_data()
            self.__draw_map0()
            self.__draw_click_point()

    def __radio3_click(self, label):
        if self.click_radio3_num == self.radio3_dict[label] and self.radio_click_flg == 1:
            self.radio_click_flg = 0
            self.click_radio3_num = -1
            for circle in self.radio3.circles:
                circle.fill = None

            self.__draw_map0()
            self.__draw_click_point()
        else:
            self.radio_click_flg = 1
            self.click_radio3_num = self.radio3_dict[label]
            for circle in self.radio3.circles:
                circle.fill = True
            for circle in self.radio1.circles:
                circle.fill = None
            for circle in self.radio2.circles:
                circle.fill = None
            self.click_radio1_num = -1
            self.click_radio2_num = -1

            self.__calc_radio_data()
            self.__draw_map0()
            self.__draw_click_point()

    # ------------------------------ #
    # -----描画--------------------- #
    # ------------------------------ #
    def draw_map(self):
        # コンポーネントの初期表示(左下が0番目のユニットが来るように行列を上下反転している)
        self.__draw_map0()
        self.__draw_map1()
        self.__draw_map2()
        self.__draw_map3()
        self.__draw_map4()
        self.__draw_click_point()

        # クリックイベント
        self.Fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)
        self.radio1.on_clicked(self.__radio1_click)
        self.radio2.on_clicked(self.__radio2_click)
        self.radio3.on_clicked(self.__radio3_click)

        # クリックイベント
        self.Fig.canvas.mpl_connect('button_press_event', self.__onclick_fig)

        # マウスオーバーイベント
        self.Fig.canvas.mpl_connect('motion_notify_event', self.__mouse_over_fig)
        self.Fig.canvas.mpl_connect('axes_leave_event', self.__mouse_leave_fig)
        plt.show()

    # ------------------------------ #
    # -----ラベルの描画-------------- #
    # ------------------------------ #
    # ユーザ
    def __draw_label0(self):
        count = np.zeros(self.Mode0_Num)
        epsilon = 0.01 * (self.Map0_position.max() - self.Map0_position.min())
        for i in range(self.Winner0.shape[0]):
            # self.Map0.plot(
            #     self.Map0_position[self.Winner0[i], 0] + epsilon * self.noise_map0[i, 0],
            #     self.Map0_position[self.Winner0[i], 1] + epsilon * self.noise_map0[i, 1],
            #     ".", color="black", ms=2)
            # self.Map0.text(self.Map0_position[self.Winner0[i], 0] + 0.7 * epsilon, self.Map0_position[self.Winner0[i], 1]
            #                - 1.1 * epsilon * count[self.Winner0[i]], i+1, va='top', size=12, color='black')
            count[self.Winner0[i]] += 1
        self.Fig.show()

    # 性格データ
    def __draw_label1(self):
        count = np.zeros(self.Mode1_Num)
        epsilon = 0.1 * (self.Map1_position.max() - self.Map1_position.min())
        for i in range(self.Winner1.shape[0]):
            self.Map1.plot(self.Map1_position[self.Winner1[i], 0], self.Map1_position[self.Winner1[i], 1],
                           ".", color="black", ms=5)

            self.Map1.text(self.Map1_position[self.Winner1[i], 0],
                           self.Map1_position[self.Winner1[i], 1] - 1. * epsilon * count[self.Winner1[i]] - 0.1,
                           self.labels1[i], ha='center', va='top', size=self.moji_size,
                           color='black', rotation=15, wrap=True)

            count[self.Winner1[i]] += 0.15
        self.Fig.show()

    # 趣味データ
    def __draw_label2(self):
        count = np.zeros(self.Mode2_Num)
        epsilon = 0.1 * (self.Map2_position.max() - self.Map2_position.min())
        for i in range(self.Winner2.shape[0]):
            self.Map2.plot(self.Map2_position[self.Winner2[i], 0], self.Map2_position[self.Winner2[i], 1],
                           ".", color="black", ms=5)

            self.Map2.text(self.Map2_position[self.Winner2[i], 0],
                           self.Map2_position[self.Winner2[i], 1] - 1. * epsilon * count[self.Winner2[i]] - 0.1,
                           self.labels2[i], ha='center', va='top', size=self.moji_size,
                           color='black', rotation=15, wrap=True)

            # self.Map2.text(self.Map2_position[self.Winner2[i], 0] + 0.7 * epsilon,
            #                self.Map2_position[self.Winner2[i], 1]
            #                - 1.1 * epsilon * count[self.Winner2[i]], i + 1, va='top', size=12, color='black')

            # self.Map1.text(self.Map1_position[self.Winner1[i], 0], self.Map1_position[self.Winner1[i], 1]
            #                + epsilon * 1.4 * count[self.Winner1[i]] + epsilon * 1.4, self.labels1[i], color='black',
            #                ha='center', va='top', bbox=self.bbox_labels)
            count[self.Winner2[i]] += 0.15
        self.Fig.show()

    # 乗り物
    def __draw_label3(self):
        count = np.zeros(self.Mode3_Num)
        epsilon = 0.1 * (self.Map3_position.max() - self.Map3_position.min())
        for i in range(self.Winner3.shape[0]):
            self.Map3.plot(self.Map3_position[self.Winner3[i], 0], self.Map3_position[self.Winner3[i], 1],
                           ".", color="black", ms=5)

            self.Map3.text(self.Map3_position[self.Winner3[i], 0],
                           self.Map3_position[self.Winner3[i], 1] - 1. * epsilon * count[self.Winner3[i]] - 0.1,
                           self.labels3[i], ha='center', va='top', size=self.moji_size,
                           color='black', rotation=15, wrap=True)
            # self.Map3.text(self.Map3_position[self.Winner3[i], 0], self.Map3_position[self.Winner3[i], 1]
            #                - epsilon * count[self.Winner3[i]], i+1, size=12, color='black')
            count[self.Winner3[i]] += 0.15
        self.Fig.show()

    # 評価
    def __draw_label4(self):
        count = np.zeros(self.Mode4_Num)
        epsilon = 0.1 * (self.Map4_position.max() - self.Map4_position.min())
        for i in range(self.Winner4.shape[0]):
            self.Map4.plot(self.Map4_position[self.Winner4[i], 0], self.Map1_position[self.Winner4[i], 1],
                           ".", color="black", ms=5)

            self.Map4.text(self.Map4_position[self.Winner4[i], 0],
                           self.Map4_position[self.Winner4[i], 1] - 1. * epsilon * count[self.Winner4[i]] - 0.1,
                           self.labels4[i], ha='center', va='top', size=self.moji_size,
                           color='black', rotation=15, wrap=True)

            # self.Map4.text(self.Map4_position[self.Winner4[i], 0] + 0.5 * epsilon,
            #                self.Map4_position[self.Winner4[i], 1]
            #                - 1.2 * epsilon * count2[self.Winner4[i]], i + 1, size=12, color='black')
            count[self.Winner4[i]] += 0.3
        self.Fig.show()
        
    # ------------------------------ #
    # -----ラベルの描画(マウスオーバ時)- #
    # ------------------------------ #
    # ユーザ
    def __draw_mouse_over_label0(self, mouse_over_unit):
        wine_labels = " "
    #     for i in range(self.Winner0.shape[0]):
    #         if mouse_over_unit == self.Winner0[i]:
    #             if len(wine_labels) <= 1:
    #                 wine_labels = self.labels0[i]
    #                 temp = i
    #             else:
    #                 wine_labels = wine_labels + "\n" + self.labels0[i]
    #     if len(wine_labels) > 1:
    #         if self.radio_click_flg == 0:
    #             self.__draw_map0()
    #         elif self.radio_click_flg == 1:
    #             self.__draw_radio1_data()
    #             self.__draw_map0()
    #         elif self.radio_click_flg == 2:
    #             self.__draw_radio2_data()
    #             self.__draw_map0()
    #         if self.Winner0[temp] % self.map0x_num < self.map0x_num/2.0:
    #             self.Map0.text(self.Map0_position[mouse_over_unit, 0], self.Map0_position[mouse_over_unit, 1],
    #                            wine_labels, color='black', ha='left', va='center', bbox=self.bbox_mouse)
    #         else:
    #             self.Map0.text(self.Map0_position[mouse_over_unit, 0], self.Map0_position[mouse_over_unit, 1],
    #                            wine_labels, color='black', ha='right', va='center', bbox=self.bbox_mouse)

    # 性格
    def __draw_mouse_over_label1(self, mouse_over_unit):
        seikaku_labels = " "
        # temp = 0
        # for i in range(self.Winner1.shape[0]):
        #     if mouse_over_unit == self.Winner1[i]:
        #         if len(seikaku_labels) <= 1:
        #             seikaku_labels = self.labels1[i]
        #             temp = i
        #         else:
        #             seikaku_labels = seikaku_labels + "\n" + self.labels1[i]
        # if len(seikaku_labels) > 1:
        #     self.__draw_map1()
        #     if self.Winner1[temp] % self.map1x_num < self.map1x_num / 2.0:
        #         self.Map1.text(self.Map1_position[mouse_over_unit, 0], self.Map1_position[mouse_over_unit, 1],
        #                        seikaku_labels, color='black', ha='left', va='center', bbox=self.bbox_mouse)
        #     else:
        #         self.Map1.text(self.Map1_position[mouse_over_unit, 0], self.Map1_position[mouse_over_unit, 1],
        #                        seikaku_labels, color='black', ha='right', va='center', bbox=self.bbox_mouse)
    
    # 趣味
    def __draw_mouse_over_label2(self, mouse_over_unit):
        syumi_labels = " "
        temp = 0
        for i in range(self.Winner2.shape[0]):
            if mouse_over_unit == self.Winner2[i]:
                if len(syumi_labels) <= 1:
                    syumi_labels = self.labels2[i]
                    temp = i
                else:
                    syumi_labels = syumi_labels + "\n" + self.labels2[i]
        if len(syumi_labels) > 1:
            self.__draw_map2()
            if self.Winner2[temp] % self.map2x_num < self.map2x_num / 2.0:
                self.Map2.text(self.Map2_position[mouse_over_unit, 0], self.Map2_position[mouse_over_unit, 1],
                               syumi_labels, color='black', ha='left', va='center', bbox=self.bbox_mouse)
            else:
                self.Map2.text(self.Map2_position[mouse_over_unit, 0], self.Map2_position[mouse_over_unit, 1],
                               syumi_labels, color='black', ha='right', va='center', bbox=self.bbox_mouse)

    # ------------------------------ #
    # -----クリック位置の描画---------- #
    # ------------------------------ #
    def __draw_click_point(self):
        # クリック位置を表示したくない場合はflgを0にする
        if self.flg == 1:
            self.Map0.plot(self.Map0_position[self.Map0_click_unit, 0], self.Map0_position[self.Map0_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Map3.plot(self.Map3_position[self.Map3_click_unit, 0], self.Map3_position[self.Map3_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            self.Map4.plot(self.Map4_position[self.Map4_click_unit, 0], self.Map4_position[self.Map4_click_unit, 1],
                           ".", color="black", ms=30, fillstyle="none")
            if self.click_map == 1:
                self.Map1.plot(self.Map1_position[self.Map1_click_unit, 0], self.Map1_position[self.Map1_click_unit, 1],
                               ".", color="black", ms=30, fillstyle="none")
            elif self.click_map == 2:
                self.Map2.plot(self.Map2_position[self.Map2_click_unit, 0], self.Map2_position[self.Map2_click_unit, 1],
                               ".", color="black", ms=30, fillstyle="none")
        self.Fig.show()

    # ------------------------------ #
    # -----コンポーネントプレーン表示-- #
    # ------------------------------ #
    # ユーザ
    def __draw_map0(self):
        self.Map0.cla()
        self.Map0.set_xlabel('ユーザマップ')
        self.__draw_label0()

        # コンポーネントを表示したくない場合はフラグを０にする
        if self.flg == 1:
            if self.radio_click_flg == 1:
                self.__calc_radio_data()
                self.Map0.imshow(self.Map0_val[::], interpolation='spline36',
                                 extent=[0, self.Map0_val.shape[0] - 1, -self.Map0_val.shape[1] + 1, 0],
                                 vmin=0, vmax=1, cmap="gray")
            else:
                self.Map0.imshow(self.Map0_val[::], interpolation='spline36', vmin=self.V_Min[self.click_map], vmax=self.V_Max[self.click_map],
                                 extent=[0, self.Map0_val.shape[0] - 1, -self.Map0_val.shape[1] + 1, 0], cmap="rainbow")
        self.Map0.set_xlim(-1, self.Mapsize)
        self.Map0.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    # 性格
    def __draw_map1(self):
        self.Map1.cla()
        # self.Map1.set_xlabel('性格')
        self.Map1.xaxis.set_label_coords(0.5, -0.1)
        self.__draw_label1()

        # コンポーネントを表示したくない場合はフラグを０にする
        if self.flg == 1:
            self.Map1.imshow(self.Map1_val[::], interpolation='spline36', vmin=self.V_Min[1], vmax=self.V_Max[1],
                             extent=[0, self.Map1_val.shape[0] - 1, -self.Map1_val.shape[1] + 1, 0], cmap="rainbow")
        self.Map1.set_xlim(-1, self.Mapsize)
        self.Map1.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    # 趣味
    def __draw_map2(self):
        self.Map2.cla()
        # self.Map2.set_xlabel('趣味')
        self.Map2.xaxis.set_label_coords(0.5, -0.1)
        self.__draw_label2()

        # コンポーネントを表示したくない場合はフラグを０にする
        if self.flg == 1:
            self.Map2.imshow(self.Map2_val[::], interpolation='spline36', vmin=self.V_Min[2], vmax=self.V_Max[2],
                             extent=[0, self.Map2_val.shape[0] - 1, -self.Map2_val.shape[1] + 1, 0], cmap="rainbow")
        self.Map2.set_xlim(-1, self.Mapsize)
        self.Map2.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    # 乗り物
    def __draw_map3(self):
        self.Map3.cla()
        # self.Map3.set_xlabel('乗り物')
        self.__draw_label3()

        # コンポーネントを表示したくない場合はフラグを０にする
        if self.flg == 1:
            self.Map3.imshow(self.Map3_val[::], interpolation='spline36', vmin=self.V_Min[3], vmax=self.V_Max[3],
                             extent=[0, self.Map3_val.shape[0] - 1, -self.Map3_val.shape[1] + 1, 0], cmap="rainbow")
        self.Map3.set_xlim(-1, self.Mapsize)
        self.Map3.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    # 評価
    def __draw_map4(self):
        self.Map4.cla()
        # self.Map4.set_xlabel('評価')
        self.Map4.xaxis.set_label_coords(0.5, -0.1)
        self.__draw_label4()

        # コンポーネントを表示したくない場合はフラグを０にする
        if self.flg == 1:
            self.Map4.imshow(self.Map4_val[::], interpolation='spline36', vmin=self.V_Min[4], vmax=self.V_Max[4],
                             extent=[0, self.Map4_val.shape[0] - 1, -self.Map4_val.shape[1] + 1, 0], cmap="rainbow")
        self.Map4.set_xlim(-1, self.Mapsize)
        self.Map4.set_ylim(-self.Mapsize, 1)
        self.Fig.show()

    # ------------------------------ #
    # -----コンポーネント値の算出------ #
    # ------------------------------ #
    # ユーザからの情報伝搬
    def __calc_propagation0(self, map_num):
        if map_num == 1:
            temp1 = self.Y1[self.Map0_click_unit, :, :]
            self.Map1_val = np.sum(temp1, axis=1).reshape([self.map1x_num, self.map1x_num])
        elif map_num == 2:
            temp2 = self.Y2[self.Map0_click_unit, :, :]
            self.Map2_val = np.sum(temp2, axis=1).reshape([self.map2x_num, self.map2x_num])
        elif map_num == 3:
            temp3 = self.Y3[self.Map0_click_unit, :, self.Map4_click_unit, :]
            self.Map3_val = np.sum(temp3, axis=1).reshape([self.map3x_num, self.map3x_num])
        elif map_num == 4:
            temp4 = self.Y3[self.Map0_click_unit, self.Map3_click_unit, :, :]
            self.Map4_val = np.sum(temp4, axis=1).reshape([self.map4x_num, self.map4x_num])

    # 性格からの情報伝播
    def __calc_propagation1(self, map_num):
        temp0 = self.Y1[:, self.Map1_click_unit, :]

        if map_num == 0:
            user_val = np.sum(temp0, axis=1)
            self.Map0_val = user_val.reshape([self.map0x_num, self.map0x_num])
        elif map_num == 2:
            temp2 = self.Y2[self.Map0_click_unit, :, :]
            self.Map2_val = np.sum(temp2, axis=1).reshape([self.map2x_num, self.map2x_num])
        elif map_num == 3:
            temp3 = self.Y3[self.Map0_click_unit, :, self.Map4_click_unit, :]
            self.Map3_val = np.sum(temp3, axis=1).reshape([self.map3x_num, self.map3x_num])
        elif map_num == 4:
            temp4 = self.Y3[self.Map0_click_unit, self.Map3_click_unit, :, :]
            self.Map4_val = np.sum(temp4, axis=1).reshape([self.map4x_num, self.map4x_num])

    # 趣味からの情報伝播
    def __calc_propagation2(self, map_num):
        temp0 = self.Y2[:, self.Map2_click_unit, :]

        if map_num == 0:
            user_val = np.sum(temp0, axis=1)
            self.Map0_val = user_val.reshape([self.map0x_num, self.map0x_num])
        elif map_num == 1:
            temp1 = self.Y1[self.Map0_click_unit, :, :]
            self.Map1_val = np.sum(temp1, axis=1).reshape([self.map1x_num, self.map1x_num])
        elif map_num == 3:
            temp3 = self.Y3[self.Map0_click_unit, :, self.Map4_click_unit, :]
            self.Map3_val = np.sum(temp3, axis=1).reshape([self.map3x_num, self.map3x_num])
        elif map_num == 4:
            temp4 = self.Y3[self.Map0_click_unit, self.Map3_click_unit, :, :]
            self.Map4_val = np.sum(temp4, axis=1).reshape([self.map4x_num, self.map4x_num])

    # 乗り物からの情報伝播
    def __calc_propagation3(self, map_num):
        if map_num == 0:
            temp0 = self.Y3[:, self.Map3_click_unit, self.Map4_click_unit, :]
            self.Map0_val = np.sum(temp0, axis=1).reshape([self.map0x_num, self.map0x_num])
        elif map_num == 1:
            temp1 = self.Y1[self.Map0_click_unit, :, :]
            self.Map1_val = np.sum(temp1, axis=1).reshape([self.map1x_num, self.map1x_num])
        elif map_num == 2:
            temp2 = self.Y2[self.Map0_click_unit, :, :]
            self.Map2_val = np.sum(temp2, axis=1).reshape([self.map2x_num, self.map2x_num])
        elif map_num == 4:
            temp4 = self.Y3[self.Map0_click_unit, self.Map3_click_unit, :, :]
            self.Map4_val = np.sum(temp4, axis=1).reshape([self.map4x_num, self.map4x_num])

    # 評価からの情報伝播
    def __calc_propagation4(self, map_num):
        if map_num == 0:
            temp0 = self.Y3[:, self.Map3_click_unit, self.Map4_click_unit, :]
            self.Map0_val = np.sum(temp0, axis=1).reshape([self.map0x_num, self.map0x_num])
        elif map_num == 1:
            temp1 = self.Y1[self.Map0_click_unit, :, :]
            self.Map1_val = np.sum(temp1, axis=1).reshape([self.map1x_num, self.map1x_num])
        elif map_num == 2:
            temp2 = self.Y2[self.Map0_click_unit, :, :]
            self.Map2_val = np.sum(temp2, axis=1).reshape([self.map2x_num, self.map2x_num])
        elif map_num == 3:
            temp3 = self.Y3[self.Map0_click_unit, :, self.Map4_click_unit, :]
            self.Map3_val = np.sum(temp3, axis=1).reshape([self.map3x_num, self.map3x_num])

    # 属性情報の表示
    def __calc_radio_data(self):
        if self.click_radio1_num != -1:
            # 性別
            temp = self.Y_attribute[:, 0]
            temp0 = np.exp(-(temp - self.click_radio1_num) ** 2)

        elif self.click_radio2_num != -1:
            # 年齢
            temp0 = self.Y_attribute[:, self.click_radio2_num + 1]

        elif self.click_radio3_num != -1:
            # 地方
            temp0 = self.Y_attribute[:, self.click_radio3_num + 8]
            print(self.click_radio3_num + 8)
            print(self.Y_attribute.shape)

        self.Map0_val = 1 - temp0.reshape([self.map0x_num, self.map0x_num])

    # ------------------------------ #
    # -----最近傍ユニット算出---------- #
    # ------------------------------ #
    @staticmethod
    def __calc_arg_min_unit(zeta, click_point):
        distance = dist.cdist(zeta, click_point)
        unit = np.argmin(distance, axis=0)
        return unit[0]
