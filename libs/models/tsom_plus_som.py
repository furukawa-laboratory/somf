from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from scipy.spatial import distance as dist


class TSOMPlusSOM:
    def __init__(self, member_features, group_features, params_tsom, params_som):
        self.params_tsom = params_tsom
        self.params_som = params_som

        self.params_tsom['X'] = member_features
        self.group_features = group_features  # グループ数の確認
        self.group_num = len(self.group_features)

    def fit(self, tsom_epoch_num, kernel_width, som_epoch_num):
        self._fit_1st_TSOM(tsom_epoch_num)
        self._fit_KDE(kernel_width)
        self._fit_2nd_SOM(som_epoch_num)

    def _fit_1st_TSOM(self, tsom_epoch_num):
        self.tsom = TSOM2(**self.params_tsom)
        self.tsom.fit(tsom_epoch_num)

    def _fit_KDE(self, kernel_width):  # 学習した後の潜在空間からKDEで確率分布を作る
        prob_data = self._calculate_kde(group_features=self.group_features,kernel_width=kernel_width)
        self.params_som['X'] = prob_data
        self.params_som['metric'] = "KLdivergence"

    def _calculate_kde(self, group_features, kernel_width):
        # グループごとにKDEを適用
        if isinstance(group_features, np.ndarray) and group_features.ndim == 2:
            # group_featuresがbag of membersで与えられた時の処理
            distance = dist.cdist(self.tsom.Zeta1, self.tsom.Z1, 'sqeuclidean')  # K1 x num_members
            H = np.exp(-0.5 * distance / (kernel_width * kernel_width))  # KxN
            prob_data = group_features @ H.T  # num_group x K1
            prob_data = prob_data / prob_data.sum(axis=1)[:, None]
        else:
            # group_featuresがlist of listsもしくはlist of arraysで与えられた時の処理
            prob_data = np.zeros((self.group_num, self.tsom.K1))  # group数*ノード数
            for i,one_group_features in enumerate(group_features):
                Dist = dist.cdist(self.tsom.Zeta1,
                                  self.tsom.Z1[one_group_features, :],
                                  'sqeuclidean')  # KxNの距離行列を計算
                H = np.exp(-Dist / (2 * kernel_width * kernel_width))  # KxNのカーネルの値を計算
                prob = np.sum(H, axis=1)
                prob_sum = np.sum(prob)
                prob = prob / prob_sum
                prob_data[i, :] = prob
        return prob_data


    def _fit_2nd_SOM(self, som_epoch_num):  # 上位のSOMを
        self.som = SOM(**self.params_som)
        self.som.fit(som_epoch_num)

    def transform(self, group_features):
        pass