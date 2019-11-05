from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from scipy.spatial import distance as dist


class TSOM_plus_SOM:
    def __init__(self, input_data, group_label, params_tsom, params_som):
        self.params_tsom = params_tsom
        self.params_som = params_som

        self.params_tsom['X'] = input_data
        self.group_label = group_label  # グループ数の確認
        self.group_num = len(self.group_label)

    def fit_1st_TSOM(self, tsom_epoch_num):
        self.tsom = TSOM2(**self.params_tsom)
        self.tsom.fit(tsom_epoch_num)

    def fit_KDE(self, kernel_width):  # 学習した後の潜在空間からKDEで確率分布を作る
        prob_data = np.zeros((self.group_num, self.tsom.K1))  # group数*ノード数
        # グループごとにKDEを適用
        for i in range(self.group_num):
            Dist = dist.cdist(self.tsom.Zeta1, self.tsom.Z1[self.group_label[i], :], 'sqeuclidean')  # KxNの距離行列を計算
            H = np.exp(-Dist / (2 * kernel_width * kernel_width))  # KxNの学習量行列を計算
            prob = np.sum(H, axis=1)
            prob_sum = np.sum(prob)
            prob = prob / prob_sum
            prob_data[i, :] = prob
        self.params_som['X'] = prob_data
        self.params_som['metric'] = "KLdivergence"

    def fit_2nd_SOM(self, som_epoch_num, init):  # 上位のSOMを
        self.params_som['init'] = init
        self.som = SOM(**self.params_som)
        self.som.fit(som_epoch_num)
