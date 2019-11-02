import numpy as np
from libs.tools.create_zeta import create_zeta
from scipy.spatial import distance as distance
from tqdm import tqdm

class TSOM3_another():
    def __init__(self, X, latent_dim, resolution, SIGMA_MAX, SIGMA_MIN, TAU, init='random'):

        # 入力データXについて
        if X.ndim == 3:
            self.X = X
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.N3=self.X.shape[2]
        else:
            raise ValueError("invalid X: {}\nX must be 3d ndarray".format(X))

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
        if type(latent_dim) is int:  # latent_dimがintであればどのモードも潜在空間の次元は同じ
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

        # K1,K2,K3は潜在空間の設定が終わった後がいいよね
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

        self.history = {}

    def fit(self, nb_epoch=200):
        self.history['y'] = np.zeros((nb_epoch, self.K1, self.K2, self.K3))
        self.history['z1'] = np.zeros((nb_epoch, self.N1, self.latent_dim1))
        self.history['z2'] = np.zeros((nb_epoch, self.N2, self.latent_dim2))
        self.history['z3'] = np.zeros((nb_epoch, self.N3, self.latent_dim3))
        self.history['sigma1'] = np.zeros(nb_epoch)
        self.history['sigma2'] = np.zeros(nb_epoch)
        self.history['sigma3'] = np.zeros(nb_epoch)

        for epoch in tqdm(np.arange(nb_epoch)):
            #-------------please write tsom3 algorithm--------------------------
            #学習量の決定
            sigma1 = max(self.SIGMA1_MIN, self.SIGMA1_MAX * (1 - (epoch / self.TAU1)))
            distance1 = distance.cdist(self.Zeta1, self.Z1, 'sqeuclidean')
            H1 = np.exp(-distance1 / (2 * pow(sigma1, 2)))
            G1 = np.sum(H1, axis=1)
            R1 = (H1.T / G1).T

            sigma2 = max(self.SIGMA2_MIN, self.SIGMA2_MAX * (1 - (epoch / self.TAU2)))
            distance2 = distance.cdist(self.Zeta2, self.Z2, 'sqeuclidean')
            H2 = np.exp(-distance2 / (2 * pow(sigma2, 2)))
            G2 = np.sum(H2, axis=1)
            R2 = (H2.T / G2).T

            sigma3 = max(self.SIGMA3_MIN, self.SIGMA3_MAX * (1 - (epoch / self.TAU3)))
            distance3 = distance.cdist(self.Zeta3, self.Z3, 'sqeuclidean')
            H3 = np.exp(-distance3 / (2 * pow(sigma3, 2)))
            G3 = np.sum(H3, axis=1)
            R3 = (H3.T / G3).T

            self.U1 = np.einsum('mj,nk,ijk->imn', R2, R3, self.X)
            self.U2 = np.einsum('li,nk,ijk->ljn', R1, R3, self.X)
            self.U3 = np.einsum('li,mj,ijk->lmk', R1, R2, self.X)
            self.Y = np.einsum('li,mj,nk,ijk->lmn', R1, R2, R3, self.X)

            self.k_star1 = np.argmin(np.sum(np.square(self.U1[:, None, :, :] - self.Y[None, :, :, :]), axis=(2, 3)), axis=1)
            self.k_star2 = np.argmin(np.sum(np.square(self.U2[:, :, None, :] - self.Y[:, None, :, :]), axis=(0, 3)), axis=1)
            self.k_star3 = np.argmin(np.sum(np.square(self.U3[:, :, :, None] - self.Y[:, :, None, :]), axis=(0, 1)), axis=1)

            self.Z1 = self.Zeta1[self.k_star1, :]
            self.Z2 = self.Zeta2[self.k_star2, :]
            self.Z3 = self.Zeta3[self.k_star3, :]

            # -------------please write tsom3 algorithm--------------------------

            self.history['y'][epoch, :, :] = self.Y
            self.history['z1'][epoch, :] = self.Z1
            self.history['z2'][epoch, :] = self.Z2
            self.history['z3'][epoch, :] = self.Z3
            self.history['sigma1'][epoch] = sigma1
            self.history['sigma2'][epoch] = sigma2
            self.history['sigma3'][epoch] = sigma3