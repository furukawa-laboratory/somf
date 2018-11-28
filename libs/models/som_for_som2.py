import numpy as np
import scipy.spatial.distance as dist

class SOM:
    def __init__(self, x, node_num = np.array([10,10]),
            sigma_max = 2.0, sigma_min = 0.2, epoch = 300, tau = 300, seed = 1):

        self.X = x
        self.X_Num = x.shape[0]
        self.X_Dim = x.shape[1]
        self.Node_Total_Num = node_num[0] * node_num[1]
        self.Node_Num = node_num
        self.Sigma_Max = sigma_max
        self.Sigma_Min = sigma_min
        self.Epoch = epoch
        self.Tau = tau
        #引数の格納

        #参照ベクトルの作成
        np.random.seed(seed)
        self.Y = np.zeros([self.Epoch + 1, self.Node_Total_Num, self.X_Dim])
        self.Y[0] = np.random.rand(self.Node_Total_Num, self.X_Dim) * 2.0 - 1.0

        #潜在空間の代表点作成
        node_x_range = np.linspace(-1, 1, self.Node_Num[0])
        node_y_range = np.linspace(-1, 1, self.Node_Num[1])
        node_x, node_y = np.meshgrid(node_x_range, node_y_range)
        self.Zeta = np.c_[node_x.ravel(), node_y.ravel()]

        #学習量
        self.H = np.zeros([self.X_Num, self.Node_Total_Num])

        #勝者ユニットの初期化
        self.BMU = np.zeros([self.Epoch, self.X_Num], dtype = int)

        #学習
    def fit(self):
        for t in range(self.Epoch):
            self.competitive_process(t)
            self.cooperative_process(t)
            self.adaptive_process(t)

        #競合過程
    def competitive_process(self, t):
        distance = dist.cdist(self.X, self.Y[t], 'sqeuclidean')
        self.BMU[t] = np.argmin(distance, axis = 1)

        #協調過程
    def cooperative_process(self, t):
        sigma = max((self.Sigma_Min - self.Sigma_Max) * t / self.Tau + self.Sigma_Max, self.Sigma_Min)
        distance_zeta = dist.cdist(self.Zeta, self.Zeta[self.BMU[t]], 'sqeuclidean')
        self.H = np.exp(-0.5 * distance_zeta / (sigma * sigma))

        #適応過程
    def adaptive_process(self, t):
        g = np.sum(self.H, axis = 1)
        self.Y[t + 1] = (self.H.T / g).T @ self.X

        #コピーバック用
    def set_y(self, y, t):
        self.Y[t] = y