import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

class TSOM2():
    def __init__(self, X,latent_dim,mode1_nodes=[10,1],mode2_nodes=[10,1],SIGMA_MAX=[2.0, 2.0] ,SIGMA_MIN=[0.2, 0.2], TAU=[50,50]):
        #パラメータの設定
        self.SIGMA1_MIN = SIGMA_MIN[0]
        self.SIGMA1_MAX = SIGMA_MAX[0]
        self.SIGMA2_MIN = SIGMA_MIN[1]
        self.SIGMA2_MAX = SIGMA_MAX[1]
        self.TAU1 = TAU[0]
        self.TAU2 = TAU[1]
        self.latent_dim=latent_dim
        self.K1 = mode1_nodes[0]*mode1_nodes[1]
        self.K2 = mode2_nodes[0]*mode2_nodes[1]
        #Xについて
        if X.ndim == 2:
            self.X = X.reshape((X.shape[0], X.shape[1], 1))
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.observed_dim = self.X.shape[2]  # 観測空間の次元

        elif X.ndim == 3:
            self.X = X
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.observed_dim = self.X.shape[2]  # 観測空間の次元
        else:
            print("X_error")

        
        #サンプル数の決定
        # self.N1 = X.shape[0]
        # self.N2 = X.shape[1]
        # self.observed_dim = X.shape[2]  # 観測空間の次元
        # self.X = X  #:N1*N2*D
        #勝者ノードの初期化
        self.Z1 = np.random.rand(self.N1, self.latent_dim)
        self.Z2 = np.random.rand(self.N2, self.latent_dim)
        self.history = {}


        #潜在空間の作成
        # if latent_dim==1:
        #     self.Zeta1 = np.linspace(-1, 1, mode1_nodes)[:,np.newaxis]
        #     self.Zeta2 = np.linspace(-1, 1, mode2_nodes)[:,np.newaxis]
        mode1_x = np.linspace(-1, 1, mode1_nodes[0])
        mode1_y = np.linspace(-1, 1, mode1_nodes[1])
        mode2_x = np.linspace(-1, 1, mode2_nodes[0])
        mode2_y = np.linspace(-1, 1, mode2_nodes[1])
        mode1_Zeta1, mode1_Zeta2 = np.meshgrid(mode1_x,mode1_y)
        mode2_Zeta1, mode2_Zeta2 = np.meshgrid(mode2_x, mode2_y)
        self.Zeta1 = np.c_[mode1_Zeta2.ravel(), mode1_Zeta1.ravel()]
        self.Zeta2 = np.c_[mode2_Zeta2.ravel(), mode2_Zeta1.ravel()]

    def fit(self,nb_epoch=200):
        self.history['y'] = np.zeros((nb_epoch, self.K1, self.K2, self.observed_dim))
        self.history['z1'] = np.zeros((nb_epoch, self.N1, self.latent_dim))
        self.history['z2'] = np.zeros((nb_epoch, self.N2, self.latent_dim))
        self.history['sigma1'] = np.zeros(nb_epoch)
        self.history['sigma2'] = np.zeros(nb_epoch)

        for epoch in tqdm(np.arange(nb_epoch)):
            # 学習量の決定
            sigma1 = self.SIGMA1_MIN + (self.SIGMA1_MAX - self.SIGMA1_MIN) * np.exp(-epoch / self.TAU1)
            distance1 = distance.cdist(self.Zeta1, self.Z1, 'sqeuclidean')  # 距離行列をつくるDはN*K行列
            H1 = np.exp(-distance1 / (2 * pow(sigma1, 2)))  # かっこに気を付ける
            G1 = np.sum(H1, axis=1)  # Gは行ごとの和をとったベクトル
            R1 = (H1.T / G1).T  # 行列の計算なので.Tで転置を行う

            sigma2 = self.SIGMA2_MIN + (self.SIGMA2_MAX - self.SIGMA2_MIN) * np.exp(-epoch / self.TAU2)
            distance2 = distance.cdist(self.Zeta2, self.Z2, 'sqeuclidean')  # 距離行列をつくるDはN*K行列
            H2 = np.exp(-distance2 / (2 * pow(sigma2, 2)))  # かっこに気を付ける
            G2 = np.sum(H2, axis=1)  # Gは行ごとの和をとったベクトル
            R2 = (H2.T / G2).T  # 行列の計算なので.Tで転置を行う
            # １次モデル，２次モデルの決定
            self.U = np.einsum('lj,ijd->ild', R2, self.X)
            self.V = np.einsum('ki,ijd->kjd', R1, self.X)
            self.Y = np.einsum('ki,lj,ijd->kld', R1, R2, self.X)
            # 勝者決定
            k_star1 = np.argmin(np.sum(np.square(self.U[:, None, :, :] - self.Y[None, :, :, :]), axis=(2, 3)), axis=1)
            k_star2 = np.argmin(np.sum(np.square(self.V[:, :, None, :] - self.Y[:, None, :, :]), axis=(0, 3)), axis=1)
            self.Z1 = self.Zeta1[k_star1, :]  # k_starのZの座標N*L(L=2
            self.Z2 = self.Zeta2[k_star2, :]  # k_starのZの座標N*L(L=2

            self.history['y'][epoch,:,:] = self.Y
            self.history['z1'][epoch,:] = self.Z1
            self.history['z2'][epoch,:] = self.Z2
            self.history['sigma1'][epoch] = sigma1
            self.history['sigma2'][epoch] = sigma2





