import numpy as np
from libs.tools.create_zeta import create_zeta
from scipy.spatial import distance as distance
from tqdm import tqdm

class wTSOM3():
    def __init__(self, X, latent_dim, resolution, SIGMA_MAX, SIGMA_MIN, TAU, gamma='nonweight', init='random'):

        # 入力データXについて
        if X.ndim == 3:
            self.X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.N3 = self.X.shape[2]
            self.observed_dim = self.X.shape[3]
        elif X.ndim == 4:
            self.X = X
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.N3 = self.X.shape[2]
            self.observed_dim = self.X.shape[3]
        else:
            raise ValueError("invalid X: {}\nX must be 3d or 4d ndarray".format(X))

        # 最大近傍半径(SIGMAX)の設定
        if type(SIGMA_MAX) is float:
            self.SIGMA1_MAX = SIGMA_MAX
            self.SIGMA2_MAX = SIGMA_MAX
            self.SIGMA3_MAX = SIGMA_MAX
        elif isinstance(SIGMA_MAX, (list, tuple)):
            self.SIGMA1_MAX = SIGMA_MAX[0]
            self.SIGMA2_MAX = SIGMA_MAX[1]
            self.SIGMA3_MAX = SIGMA_MAX[2]
        else:
            raise ValueError("invalid SIGMA_MAX: {}".format(SIGMA_MAX))

        # 最小近傍半径(SIGMA_MIN)の設定
        if type(SIGMA_MIN) is float:
            self.SIGMA1_MIN = SIGMA_MIN
            self.SIGMA2_MIN = SIGMA_MIN
            self.SIGMA3_MIN = SIGMA_MIN
        elif isinstance(SIGMA_MIN, (list, tuple)):
            self.SIGMA1_MIN = SIGMA_MIN[0]
            self.SIGMA2_MIN = SIGMA_MIN[1]
            self.SIGMA3_MIN = SIGMA_MIN[2]
        else:
            raise ValueError("invalid SIGMA_MIN: {}".format(SIGMA_MIN))

        # 時定数(TAU)の設定
        if type(TAU) is int:
            self.TAU1 = TAU
            self.TAU2 = TAU
            self.TAU3 = TAU
        elif isinstance(TAU, (list, tuple)):
            self.TAU1 = TAU[0]
            self.TAU2 = TAU[1]
            self.TAU3 = TAU[2]
        else:
            raise ValueError("invalid TAU: {}".format(TAU))

        # resolutionの設定
        if type(resolution) is int:
            resolution1 = resolution
            resolution2 = resolution
            resolution3 = resolution
        elif isinstance(resolution, (list, tuple)):
            resolution1 = resolution[0]
            resolution2 = resolution[1]
            resolution3 = resolution[2]
        else:
            raise ValueError("invalid resolution: {}".format(resolution))

        # 潜在空間の設定
        if type(latent_dim) is int:  # latent_dimがintであればどちらのモードも潜在空間の次元は同じ
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim
            self.latent_dim3 = latent_dim

        elif isinstance(latent_dim, (list, tuple)):
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]
            self.latent_dim3 = latent_dim[2]
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))
        self.Zeta1 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim1, resolution=resolution1, include_min_max=True)
        self.Zeta2 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim2, resolution=resolution2, include_min_max=True)
        self.Zeta3 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim3, resolution=resolution3, include_min_max=True)

        # K1とK2は潜在空間の設定が終わった後がいいよね
        self.K1 = self.Zeta1.shape[0]
        self.K2 = self.Zeta2.shape[0]
        self.K3 = self.Zeta3.shape[0]

        # 勝者ノードの初期化
        self.Z1 = None
        self.Z2 = None
        self.Z3 = None
        if isinstance(init, str) and init in 'random':
            self.Z1 = np.random.rand(self.N1, self.latent_dim1) * 2.0 - 1.0
            self.Z2 = np.random.rand(self.N2, self.latent_dim2) * 2.0 - 1.0
            self.Z3 = np.random.rand(self.N3, self.latent_dim3) * 2.0 - 1.0
        elif isinstance(init, (tuple, list)) and len(init) == 3:
            if isinstance(init[0], np.ndarray) and init[0].shape == (self.N1, self.latent_dim1):
                self.Z1 = init[0].copy()
            else:
                raise ValueError("invalid inits[0]: {}".format(init))
            if isinstance(init[1], np.ndarray) and init[1].shape == (self.N2, self.latent_dim2):
                self.Z2 = init[1].copy()
            else:
                raise ValueError("invalid inits[1]: {}".format(init))
            if isinstance(init[2], np.ndarray) and init[2].shape == (self.N3, self.latent_dim3):
                self.Z3 = init[2].copy()
            else:
                raise ValueError("invalid inits[2]: {}".format(init))
        else:
            raise ValueError("invalid inits: {}".format(init))

        # 重みテンソルの初期化
        self.gamma = None
        if isinstance(gamma, str) and gamma in 'nonweight':
            self.gamma = np.ones((self.N1, self.N2, self.N3))
        # elif isinstance(gamma, (tuple, list)) and gamma.ndim == 3:
        #     self.gamma = gamma
        # else:
        #     raise ValueError("invalid gamma: {}".format(gamma))
        else:
            self.gamma = gamma

        self.history = {}

        # デバッグ用
        self.H1 = None
        self.H2 = None
        self.H3 = None
        
        self.G1 = None
        self.G2 = None
        self.G3 = None

        self.k_star1 = None
        self.k_star2 = None
        self.k_star3 = None
        

    def fit(self, nb_epoch=200):
        self.history['y'] = np.zeros((nb_epoch, self.K1, self.K2, self.K3, self.observed_dim))
        self.history['z1'] = np.zeros((nb_epoch, self.N1, self.latent_dim1))
        self.history['z2'] = np.zeros((nb_epoch, self.N2, self.latent_dim2))
        self.history['z3'] = np.zeros((nb_epoch, self.N3, self.latent_dim3))
        self.history['sigma1'] = np.zeros(nb_epoch)
        self.history['sigma2'] = np.zeros(nb_epoch)
        self.history['sigma3'] = np.zeros(nb_epoch)

        for epoch in tqdm(np.arange(nb_epoch)):
            # 学習量の決定
            sigma1 = max(self.SIGMA1_MIN, self.SIGMA1_MAX * (1 - (epoch / self.TAU1)))
            distance1 = distance.cdist(self.Zeta1, self.Z1, 'sqeuclidean')  # 距離行列をつくるDはK1*N1行列
            H1 = np.exp(-0.5*distance1 / (2 * pow(sigma1, 2)))  # かっこに気を付ける
            self.H1 = np.exp(-0.5*distance1 / (2 * pow(sigma1, 2)))

            sigma2 = max(self.SIGMA2_MIN, self.SIGMA2_MAX * (1 - (epoch / self.TAU2)))
            distance2 = distance.cdist(self.Zeta2, self.Z2, 'sqeuclidean')  # 距離行列をつくるDはK2*N2行列
            H2 = np.exp(-0.5*distance2 / (2 * pow(sigma2, 2)))  # かっこに気を付ける
            self.H2 = np.exp(-0.5*distance2 / (2 * pow(sigma2, 2)))

            sigma3 = max(self.SIGMA3_MIN, self.SIGMA3_MAX * (1 - (epoch / self.TAU3)))
            distance3 = distance.cdist(self.Zeta3, self.Z3, 'sqeuclidean')  # 距離行列をつくるDはK3*N3行列
            H3 = np.exp(-0.5*distance3 / (2 * pow(sigma3, 2)))  # かっこに気を付ける
            self.H3 = np.exp(-0.5*distance3 / (2 * pow(sigma3, 2)))

            G1 = np.sum(
                np.sum(self.gamma[:, None, :, :] * H2[None, :, :, None], axis=2)[:, :, None, :] * H3[None, None, :, :],
                axis=3)  # Gは行ごとの重みと掛け合わせて和をとったテンソル n1*k2*k3
            # G1 = np.einsum('ijk,jm,kn->imn',self.gamma,H2,H3)
            G2 = np.sum(
                np.sum(self.gamma[None, :, :, :] * H1[:, :, None, None], axis=1)[:, :, None, :] * H3[None, None, :, :],
                axis=3)  # Gは行ごとの重みと掛け合わせて和をとったテンソル k1*n2*k3
            # G2 = np.einsum('ijk,il,kn->ljn',self.gamma,H1,H3)
            G3 = np.sum(
                np.sum(self.gamma[None, :, :, :] * H1[:, :, None, None], axis=1)[:, None, :, :] * H2[None, :, :, None],
                axis=2)  # Gは行ごとの重みと掛け合わせて和をとったテンソル k1*k2*n3
            # G3 = np.einsum('ijk,il,jm->lmk',self.gamma,H1,H2)
            # G4 = np.einsum('ijk,il,jm,kn->lmn',self.gamma,H1,H2,H3)

            # １次モデル，２次モデルの決定
            # データ: i,j,k
            # ノード: l,m,n
            self.U1 = np.sum(
                H2[None, :, :, None, None] * np.sum(H3[None, None, :, :, None] * (self.gamma[:, :, :, None] * self.X
                )[:, :, None, :, :], axis=3)[:, None, :, :, :], axis=2) / G1[:, :, :, None]  # N1*K2*K3*D
            # self.U1 = np.einsum('ijk,jm,kn,ijkd->imnd',self.gamma,H2,H3,self.X)/G1[:,:,:,None]
            self.U2 = np.sum(
                H1[:, :, None, None, None] * np.sum(H3[None, None, :, :, None] * (self.gamma[:, :, :, None] * self.X
                )[:, :, None, :, :], axis=3)[None, :, :, :, :], axis=1) / G2[:, :, :, None]  # K1*N2*K3*D
            # self.U2 = np.einsum('ijk,il,kn,ijkd->ljnd',self.gamma,H1,H3,self.X)/G2[:,:,:,None]
            self.U3 = np.sum(
                H1[:, :, None, None, None] * np.sum(H2[None, :, :, None, None] * (self.gamma[:, :, :, None] * self.X
                )[:, None, :, :, :], axis=2)[None, :, :, :, :], axis=1) / G3[:, :, :, None]  # K1*K2*N3
            # self.U3 = np.einsum('ijk,il,jm,ijkd->lmkd',self.gamma,H1,H2,self.X)/G3[:,:,:,None]
            # １次モデルを使って2次モデルを更新
            self.Y = np.sum(H1[:, :, None, None, None] * (G1[:, :, :, None] * self.U1)[None, :, :, :, :],
                            axis=1) / np.sum(H1[:, :, None, None] * G1[None, :, :, :], axis=1)[:, :, :, None] # K1*K2*K3*D
            # self.Y = np.sum((H1[:, :, None, None] * G1[:, None, :, :])[:, :, :, :, None] * self.U1[:, None, :, :, :],
            #                 axis=0) / np.sum(H1[:, :, None, None] * G1[:, None, :, :], axis=0)[:, :, :, None]  # K1*K2*K3*D
            # self.Y = np.sum(np.sum(
            #     np.sum((self.gamma[:, :, :, None] * self.X)[:, None, :, :, :] * H1[:, :, None, None, None], axis=0)[:,
            #     :, None, :, :] * H2[None, :, :, None, None], axis=1)[:, :, :, None, :] * H3[None, None, :, :, None],
            #                 axis=2) / G4[:, :, :, None]
            # self.Y = np.einsum('ijk,il,jm,kn,ijkd->lmnd',self.gamma,H1,H2,H3,self.X)  # K1*K2*K3*D

            # 勝者決定
            # モード1
            Dist1 = np.square(self.U1[None, :, :, :, :] - self.Y[:, None, :, :, :])  # N1*K1*K2*K3*D
            Dist1_sum = np.sum(G1[None, :, :, :, None] * Dist1, axis=(2, 3, 4))  # K1*N1
            self.k_star1 = np.argmin(Dist1_sum, axis=0)  # N1*1
            self.Z1 = self.Zeta1[self.k_star1, :]

            # モード2
            Dist2 = np.square(self.U2[:, None, :, :, :] - self.Y[:, :, None, :, :])  # K1*N2*K2*K3*D
            Dist2_sum = np.sum(G2[:, None, :, :, None] * Dist2, axis=(0, 3, 4))  # K2*N2
            self.k_star2 = np.argmin(Dist2_sum, axis=0)
            self.Z2 = self.Zeta2[self.k_star2, :]

            # モード3
            Dist3 = np.square(self.U3[:, :, None, :, :] - self.Y[:, :, :, None, :])  # K1*K2*N3*K3*D
            Dist3_sum = np.sum(G3[:, :, None, :, None] * Dist3, axis=(0, 1, 4))  # K3*N3
            self.k_star3 = np.argmin(Dist3_sum, axis=0)  # N3*1
            self.Z3 = self.Zeta3[self.k_star3, :]

            self.history['y'][epoch, :, :, :, :] = self.Y
            self.history['z1'][epoch, :] = self.Z1
            self.history['z2'][epoch, :] = self.Z2
            self.history['z3'][epoch, :] = self.Z3
            self.history['sigma1'][epoch] = sigma1
            self.history['sigma2'][epoch] = sigma2
            self.history['sigma3'][epoch] = sigma3
