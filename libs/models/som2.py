import numpy as np
import sys
sys.path.append('./')
from libs.models.som_for_som2 import SOM
import scipy.spatial.distance as dist


class SOM2:
    def __init__(self, data, parent_node_num=np.array([7, 1]), child_node_num=np.array([10, 10]),
                 sigma_max=[2.0, 2.0], sigma_min=[0.3, 0.1], epoch=300, tau=[200, 300], seed=1):
        # データの格納
        self.Data = data
        self.Class_Num = data.shape[0]
        self.Data_Num = data.shape[1]
        self.Data_Dim = data.shape[2]

        # SOMのパラメータ格納
        self.pNode_Total_Num = parent_node_num[0] * parent_node_num[1]
        self.pNode_Num = parent_node_num
        self.cNode_Total_Num = child_node_num[0] * child_node_num[1]
        self.cNode_Num = child_node_num
        self.Sigma_Max = sigma_max
        self.Sigma_Min = sigma_min
        self.Epoch = epoch
        self.Tau = tau

        # 親SOMへのデータ
        self.X = np.zeros([self.Class_Num, self.cNode_Total_Num*self.Data_Dim])
        self.X_Num = self.X.shape[0]
        self.X_Dim = self.X.shape[1]

        # 参照ベクトルの作成
        np.random.seed(seed)
        self.Y = np.zeros([self.Epoch+1, self.pNode_Total_Num, self.X_Dim])
        self.Y[0] = np.random.rand(self.pNode_Total_Num, self.X_Dim) * 2.0 - 1.0

        # 潜在空間の代表点作成
        node_x_range = np.linspace(-1, 1, self.pNode_Num[0])
        node_y_range = np.linspace(-1, 1, self.pNode_Num[1])
        node_x, node_y = np.meshgrid(node_x_range, node_y_range)
        self.Zeta = np.c_[node_x.ravel(), node_y.ravel()]

        # 学習量
        self.H = np.zeros([self.Class_Num, self.pNode_Total_Num])

        # 勝者ユニットの初期化
        self.BMU = np.zeros([self.Epoch, self.Class_Num], dtype=int)

        # 子SOMの宣言
        self.Child_SOM = []
        for i in range(self.Class_Num):
            self.Child_SOM.append(SOM(self.Data[i], node_num=self.cNode_Num, sigma_max=self.Sigma_Max[1],
                                          sigma_min=self.Sigma_Min[1], epoch=self.Epoch, tau=self.Tau[1]))

    # 学習
    def fit(self):
        for t in range(self.Epoch):
            # 子SOMの学習
            for i in range(self.Class_Num):
                self.Child_SOM[i].competitive_process(t)
                self.Child_SOM[i].cooperative_process(t)
                self.Child_SOM[i].adaptive_process(t)

            # 子SOMの学習結果を親SOMにInput
            for i in range(self.Class_Num):
                self.set_x(self.Child_SOM[i].Y[t+1], i)

            # 親SOMの学習
            self.competitive_process(t)
            self.cooperative_process(t)
            self.adaptive_process(t)

            # 親SOMの学習結果を子SOMにコピーバック
            for i in range(self.Class_Num):
                self.Child_SOM[i].set_y(self.Y[t+1][self.BMU[t][i]].reshape(self.cNode_Total_Num, self.Data_Dim), t+1)

    # 親SOMに渡す用
    def set_x(self, x, i):
        self.X[i] = x.reshape(self.X_Dim)

    # 競合過程
    def competitive_process(self, t):
        distance = dist.cdist(self.X, self.Y[t], 'sqeuclidean')
        self.BMU[t] = np.argmin(distance, axis=1)

    # 協調過程
    def cooperative_process(self, t):
        sigma = max((self.Sigma_Min[0] - self.Sigma_Max[0]) * t / self.Tau[0] + self.Sigma_Max[0], self.Sigma_Min[0])
        distance_zeta = dist.cdist(self.Zeta, self.Zeta[self.BMU[t]], 'sqeuclidean')
        self.H = np.exp(-0.5 * distance_zeta / (sigma * sigma))

    # 適応過程
    def adaptive_process(self, t):
        g = np.sum(self.H, axis=1)
        self.Y[t + 1] = (self.H.T / g).T @ self.X


