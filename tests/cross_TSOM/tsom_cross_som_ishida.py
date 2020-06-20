from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from scipy.spatial import distance as dist


class TSOMCrossSOM:
    def __init__(self, datasets, latent_dim,resolution,SIGMA_MAX, SIGMA_MIN, TAU,init):

        # #データセットに関しての例外処理
        # if datasets.ndim == 2:
        #     self.X = X.reshape((X.shape[0], X.shape[1], 1))
        #     self.N1 = self.X.shape[0]
        #     self.N2 = self.X.shape[1]
        #     self.observed_dim = self.X.shape[2]  # 観測空間の次元
        #
        # elif X.ndim == 3:
        #     self.X = X
        #     self.N1 = self.X.shape[0]
        #     self.N2 = self.X.shape[1]
        #     self.observed_dim = self.X.shape[2]  # 観測空間の次元
        # else:
        #     raise ValueError("invalid X: {}\nX must be 2d or 3d ndarray".format(X))


        self.datasets=datasets#listかndarrayで与えるか
        self.class_num=datasets.shape[0]
        self.N1=datasets.shape[1]
        self.N2=datasets.shape[2]
        self.observed_dim=datasets.shape[3]

        self.child_latent_dim=latent_dim#子供と親
        self.child_resolution=resolution#子供と親
        self.child_sigma_max=SIGMA_MAX#各子TSOMと親





    def fit(self, tsom_epoch_num, kernel_width, som_epoch_num):
        # 下位TSOMの定義
        child_TSOM = []

        for i in np.arange(self.class_num):
            temp_class = TSOM2(X=self.datasets[i],latent_dim=self.child_latent_dim,resolution=self.child_resolution,SIGMA_MAX=self.child_sigma_max)

    #     self._fit_1st_TSOM(tsom_epoch_num)
    #     self._fit_KDE(kernel_width)
    #     self._fit_2nd_SOM(som_epoch_num)
    #
    # def _fit_1st_TSOM(self, tsom_epoch_num):
    #     self.tsom = TSOM2(**self.params_tsom)
    #     self.tsom.fit(tsom_epoch_num)
    #
    # def _fit_KDE(self, kernel_width):  # 学習した後の潜在空間からKDEで確率分布を作る
    #     prob_data = np.zeros((self.group_num, self.tsom.K1))  # group数*ノード数
    #     # グループごとにKDEを適用
    #     for i in range(self.group_num):
    #         Dist = dist.cdist(self.tsom.Zeta1, self.tsom.Z1[self.index_members_of_group[i], :],
    #                           'sqeuclidean')  # KxNの距離行列を計算
    #         H = np.exp(-Dist / (2 * kernel_width * kernel_width))  # KxNの学習量行列を計算
    #         prob = np.sum(H, axis=1)
    #         prob_sum = np.sum(prob)
    #         prob = prob / prob_sum
    #         prob_data[i, :] = prob
    #     self.params_som['X'] = prob_data
    #     self.params_som['metric'] = "KLdivergence"
    #
    # def _fit_2nd_SOM(self, som_epoch_num):  # 上位のSOMを
    #     self.som = SOM(**self.params_som)
    #     self.som.fit(som_epoch_num)
