import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from libs.tools.create_zeta import create_zeta

class TSOM2():
    def __init__(self, X, latent_dim, resolution, SIGMA_MAX, SIGMA_MIN, TAU, init='random'):

        #最大近傍半径(SIGMAX)の設定
        if type(SIGMA_MAX) is float:
            self.SIGMA1_MAX = SIGMA_MAX
            self.SIGMA2_MAX = SIGMA_MAX
        elif type(SIGMA_MAX) is tuple:
            self.SIGMA1_MAX=SIGMA_MAX[0]
            self.SIGMA2_MAX=SIGMA_MAX[1]
        else:
            raise ValueError("invalid SIGMA_MAX: {}".format(SIGMA_MAX))

        # 最小近傍半径(SIGMA_MIN)の設定
        if type(SIGMA_MIN) is float:
            self.SIGMA1_MIN = SIGMA_MIN
            self.SIGMA2_MIN = SIGMA_MIN
        elif type(SIGMA_MIN) is tuple:
            self.SIGMA1_MIN=SIGMA_MIN[0]
            self.SIGMA2_MIN=SIGMA_MIN[1]
        else:
            raise ValueError("invalid SIGMA_MIN: {}".format(SIGMA_MIN))

        # 時定数(TAU)の設定
        if type(TAU) is int:
            self.TAU1 = TAU
            self.TAU1 = TAU
        elif type(TAU) is tuple:
            self.TAU1=TAU[0]
            self.TAU2=TAU[1]
        else:
            raise ValueError("invalid TAU: {}".format(TAU))

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
            print("X please 2mode tensor or 3 mode tensor")
            raise ValueError("invalid X: {}".format(X))

        #resolutionの設定
        if type(resolution) is int:
            resolution1=resolution
            resolution2=resolution
        elif type(resolution) is tuple:
            resolution1=resolution[0]
            resolution2=resolution[1]
        else:
            print("please tuple or int")

        #潜在空間の設定
        if type(latent_dim) is int:#latent_dimがintであればどちらのモードも潜在空間の次元は同じ
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim

        elif type(latent_dim) is tuple:#latent_dimがtupleであれば各モードで潜在空間の次元を決定
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]
        else:
            print("latent_dim please int or tuple")
            #latent_dimがlist,float,3次元以上はエラーかな?
        self.Zeta1 = create_zeta(-1.0,1.0,latent_dim=self.latent_dim1,resolution=resolution1,include_min_max=True)
        self.Zeta2 = create_zeta(-1.0,1.0,latent_dim=self.latent_dim2,resolution=resolution2,include_min_max=True)

        #K1とK2は潜在空間の設定が終わった後がいいよね
        self.K1 = self.Zeta1.shape[0]
        self.K2 = self.Zeta2.shape[0]

        #勝者ノードの初期化
        self.Z1 = None
        self.Z2 = None
        if isinstance(init, str) and init in 'random':
            self.Z1 = np.random.rand(self.N1, self.latent_dim1) * 2.0 - 1.0
            self.Z2 = np.random.rand(self.N2, self.latent_dim2) * 2.0 - 1.0
        elif isinstance(init, tuple) and len(init) == 2:
            if isinstance(init[0], np.ndarray) and init[0].shape == (self.N1, self.latent_dim1):
                self.Z1 = init[0].copy()
            else:
                raise ValueError("invalid inits[0]: {}".format(init))
            if isinstance(init[1], np.ndarray) and init[1].shape == (self.N2, self.latent_dim2):
                self.Z2 = init[1].copy()
            else:
                raise ValueError("invalid inits[1]: {}".format(init))
        else:
            raise ValueError("invalid inits: {}".format(init))

        self.history = {}


    def fit(self,nb_epoch=200):
        self.history['y'] = np.zeros((nb_epoch, self.K1, self.K2, self.observed_dim))
        self.history['z1'] = np.zeros((nb_epoch, self.N1, self.latent_dim1))
        self.history['z2'] = np.zeros((nb_epoch, self.N2, self.latent_dim2))
        self.history['sigma1'] = np.zeros(nb_epoch)
        self.history['sigma2'] = np.zeros(nb_epoch)

        for epoch in tqdm(np.arange(nb_epoch)):
            # 学習量の決定
            #sigma1 = self.SIGMA1_MIN + (self.SIGMA1_MAX - self.SIGMA1_MIN) * np.exp(-epoch / self.TAU1)
            sigma1 = max(self.SIGMA1_MIN, self.SIGMA1_MAX * ( 1 - (epoch / self.TAU1) ) )
            distance1 = distance.cdist(self.Zeta1, self.Z1, 'sqeuclidean')  # 距離行列をつくるDはN*K行列
            H1 = np.exp(-distance1 / (2 * pow(sigma1, 2)))  # かっこに気を付ける
            G1 = np.sum(H1, axis=1)  # Gは行ごとの和をとったベクトル
            R1 = (H1.T / G1).T  # 行列の計算なので.Tで転置を行う

            #sigma2 = self.SIGMA2_MIN + (self.SIGMA2_MAX - self.SIGMA2_MIN) * np.exp(-epoch / self.TAU2)
            sigma2 = max(self.SIGMA2_MIN, self.SIGMA2_MAX * ( 1 - (epoch / self.TAU2) ) )
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





