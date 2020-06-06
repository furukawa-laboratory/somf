import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

from ..tools.create_zeta import create_zeta

class TSOM2():
    def __init__(self, X, latent_dim, resolution, SIGMA_MAX, SIGMA_MIN, TAU, model=None, gamma=None, init='random'):

        # 入力データXについて
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
            raise ValueError("invalid X: {}\nX must be 2d or 3d ndarray".format(X))

        if gamma is not None:  # gammaが指定されている時
            # 欠損値アルゴリズム処理
            if X.shape != gamma.shape:
                raise ValueError("invalid gamma: {}\ndata size and gamma size is not match. ".format(gamma))

            elif X.shape == gamma.shape:
                if np.any(np.isnan(self.X)) == 1:  # gamma指定してデータに欠損がある場合
                    temp_gamma = np.where(np.isnan(self.X) == 1, 0, 1)  # データに基づいてgammaを作る
                    temp_is_missing = np.allclose(temp_gamma, gamma)
                    self.X[np.isnan(self.X)] = 0  # 欠損値の部分を0で置換
                    if temp_is_missing is True:  # データの欠損しているところとgammaの0の値が一致する時
                        self.gamma = gamma
                        self.is_missing = 1
                    else:
                        raise ValueError("invalid gamma: {}\ndata size and gamma size is not match. ".format(gamma))
                elif np.any(np.isnan(self.X)) == 0:  # 観測データの一部を無視したい時
                    self.gamma = gamma
                    self.is_missing = 1
        elif gamma is None:#データXに欠損がある場合はそれに基づいてgammaを作成する
            self.is_missing=np.any(np.isnan(self.X))# 欠損値があるかを判定.欠損があれば1,欠損がなければ0
            # 欠損値がある場合
            if self.is_missing == 1:
                gamma = np.where(np.isnan(self.X) == 1, 0, 1)#nan格納されているindexを返す
                self.gamma = gamma
                self.X[np.isnan(self.X)] = 0#欠損値の部分を0で置換
            elif self.is_missing==0:#欠損値がない場合はgammaは作らない
                pass

        # 1次モデル型と直接型を選択する引数
        if model=="direct":
            self.model = "direct"
        elif model==None or model=="indirect":
            self.model="indirect"
        else:
            raise ValueError("invalid model: {}\nmodel is only direct or indirect. ".format(model))

        # 最大近傍半径(SIGMAX)の設定
        if type(SIGMA_MAX) is float:
            self.SIGMA1_MAX = SIGMA_MAX
            self.SIGMA2_MAX = SIGMA_MAX
        elif isinstance(SIGMA_MAX, (list, tuple)):
            self.SIGMA1_MAX = SIGMA_MAX[0]
            self.SIGMA2_MAX = SIGMA_MAX[1]
        else:
            raise ValueError("invalid SIGMA_MAX: {}".format(SIGMA_MAX))

        # 最小近傍半径(SIGMA_MIN)の設定
        if type(SIGMA_MIN) is float:
            self.SIGMA1_MIN = SIGMA_MIN
            self.SIGMA2_MIN = SIGMA_MIN
        elif isinstance(SIGMA_MIN, (list, tuple)):
            self.SIGMA1_MIN = SIGMA_MIN[0]
            self.SIGMA2_MIN = SIGMA_MIN[1]
        else:
            raise ValueError("invalid SIGMA_MIN: {}".format(SIGMA_MIN))

        # 時定数(TAU)の設定
        if type(TAU) is int:
            self.TAU1 = TAU
            self.TAU2 = TAU
        elif isinstance(TAU, (list, tuple)):
            self.TAU1 = TAU[0]
            self.TAU2 = TAU[1]
        else:
            raise ValueError("invalid TAU: {}".format(TAU))

        # resolutionの設定
        if type(resolution) is int:
            resolution1 = resolution
            resolution2 = resolution
        elif isinstance(resolution, (list, tuple)):
            resolution1 = resolution[0]
            resolution2 = resolution[1]
        else:
            raise ValueError("invalid resolution: {}".format(resolution))

        # 潜在空間の設定
        if type(latent_dim) is int:
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim

        elif isinstance(latent_dim, (list, tuple)):
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))
            # latent_dimがlist,float,3次元以上はエラーかな?
        self.Zeta1 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim1, resolution=resolution1, include_min_max=True)
        self.Zeta2 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim2, resolution=resolution2, include_min_max=True)

        self.K1 = self.Zeta1.shape[0]
        self.K2 = self.Zeta2.shape[0]

        # 勝者ノードの初期化
        self.Z1 = None
        self.Z2 = None
        if isinstance(init, str) and init in 'random':
            self.Z1 = np.random.rand(self.N1, self.latent_dim1) * 2.0 - 1.0
            self.Z2 = np.random.rand(self.N2, self.latent_dim2) * 2.0 - 1.0
        elif isinstance(init, (tuple, list)) and len(init) == 2:
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

    # 協調過程
    def _cooperative_process(self, epoch):
        # 学習量の決定
        # self.sigma1 = self.SIGMA1_MIN + (self.SIGMA1_MAX - self.SIGMA1_MIN) * np.exp(-epoch / self.TAU1)
        self.sigma1 = max(self.SIGMA1_MIN, self.SIGMA1_MAX * (1 - (epoch / self.TAU1)))
        distance1 = distance.cdist(self.Zeta1, self.Z1, 'sqeuclidean')  # 距離行列をつくるDはN*K行列
        H1 = np.exp(-distance1 / (2 * pow(self.sigma1, 2)))  # かっこに気を付ける
        G1 = np.sum(H1, axis=1)  # Gは行ごとの和をとったベクトル
        self.R1 = (H1.T / G1).T  # 行列の計算なので.Tで転置を行う

        # self.sigma2 = self.SIGMA2_MIN + (self.SIGMA2_MAX - self.SIGMA2_MIN) * np.exp(-epoch / self.TAU2)
        self.sigma2 = max(self.SIGMA2_MIN, self.SIGMA2_MAX * (1 - (epoch / self.TAU2)))
        distance2 = distance.cdist(self.Zeta2, self.Z2, 'sqeuclidean')  # 距離行列をつくるDはN*K行列
        H2 = np.exp(-distance2 / (2 * pow(self.sigma2, 2)))  # かっこに気を付ける
        G2 = np.sum(H2, axis=1)  # Gは行ごとの和をとったベクトル
        self.R2 = (H2.T / G2).T  # 行列の計算なので.Tで転置を行う
    
    # 適応過程
    def _adaptive_process_missing_indirect(self):
        # １次モデル，２次モデルの決定
        self.U = np.einsum('jl,ijd,ijd->ild', H2.T, self.gamma, self.X) / np.einsum('ijd,jl->ild', self.gamma, H2.T)
        self.V = np.einsum('ik,ijd,ijd->kjd', H1.T, self.gamma, self.X) / np.einsum('ijd,ik->kjd', self.gamma, H1.T)
        G = np.einsum("ik,jl,ijd->kld", H1.T, H2.T, self.gamma)  # K1*K2*D
        self.Y = np.einsum('ik,jl,ijd,ijd->kld', H1.T, H2.T, self.gamma, self.X) / G

    def _adaptive_process_missing_direct(self):
        G = np.einsum("ik,jl,ijd->kld", H1.T, H2.T, self.gamma)  # K1*K2*D
        self.Y = np.einsum('ik,jl,ijd,ijd->kld', H1.T, H2.T, self.gamma, self.X) / G

    def _adaptive_process_nonmissing_indirect(self):
        # １次モデル，２次モデルの決定
        self.U = np.einsum('lj,ijd->ild', self.R2, self.X)
        self.V = np.einsum('ki,ijd->kjd', self.R1, self.X)
        # ２次モデルの決定
        self.Y = np.einsum('ki,lj,ijd->kld', self.R1, self.R2, self.X)

    def _adaptive_process_nonmissing_direct(self):
        # 直接型
        self.Y = np.einsum('ki,lj,ijd->kld', self.R1, self.R2, self.X)

    # 競合過程
    def _competitive_process_missing_indirect(self):
        # 勝者決定
        self.k_star1 = np.argmin(
            np.sum(np.square(self.U[:, None, :, :] - self.Y[None, :, :, :]), axis=(2, 3)), axis=1)
        self.k_star2 = np.argmin(
            np.sum(np.square(self.V[:, :, None, :] - self.Y[:, None, :, :]), axis=(0, 3)), axis=1)

    def _competitive_process_missing_direct(self):
        # 勝者決定
        Dist = self.gamma[:, :, None, None, :] * np.square(
            self.X[:, :, None, None, :] - self.Y[None, None, :, :, :])
        self.k_star1 = np.argmin(np.einsum("jl,ijklm->ik", H2.T, Dist), axis=1)
        self.k_star2 = np.argmin(np.einsum("ik,ijklm->jl", H1.T, Dist), axis=1)

    def _competitive_process_nonmissing_indirect(self):
        # 勝者決定
        self.k_star1 = np.argmin(
            np.sum(np.square(self.U[:, None, :, :] - self.Y[None, :, :, :]), axis=(2, 3)), axis=1)
        self.k_star2 = np.argmin(
            np.sum(np.square(self.V[:, :, None, :] - self.Y[:, None, :, :]), axis=(0, 3)), axis=1)

    def _competitive_process_nonmissing_direct(self):
        # 勝者決定
        Dist = np.square(
            self.X[:, :, None, None, :] - self.Y[None, None, :, :, :])
        self.k_star1 = np.argmin(np.einsum("jl,ijklm->ik", H2.T, Dist), axis=1)
        self.k_star2 = np.argmin(np.einsum("ik,ijklm->jl", H1.T, Dist), axis=1)


    def fit(self, nb_epoch=200):
        self.history['y'] = np.zeros((nb_epoch, self.K1, self.K2, self.observed_dim))
        self.history['z1'] = np.zeros((nb_epoch, self.N1, self.latent_dim1))
        self.history['z2'] = np.zeros((nb_epoch, self.N2, self.latent_dim2))
        self.history['self.sigma1'] = np.zeros(nb_epoch)
        self.history['self.sigma2'] = np.zeros(nb_epoch)

        for epoch in tqdm(np.arange(nb_epoch)):

            if self.is_missing == 1: # 欠損値有り
                if self.model == "indirect":
                    self._cooperative_process(epoch)
                    self._adaptive_process_missing_indirect()
                    self._competitive_process_missing_indirect()
                elif self.model == "direct":
                    self._cooperative_process(epoch)
                    self._adaptive_process_missing_direct()
                    self._competitive_process_missing_direct()
                else:
                    raise ValueError("invalid model: {}\nmodel must be None or direct".format(self.model))

            else: # 欠損値無し
                if self.model == "indirect":
                    self._cooperative_process(epoch)
                    self._adaptive_process_nonmissing_indirect()
                    self._competitive_process_nonmissing_indirect()
                elif self.model == "direct":
                    self._cooperative_process(epoch)
                    self._adaptive_process_nonmissing_direct()
                    self._competitive_process_nonmissing_direct()
                else:
                    raise ValueError("invalid model: {}\nmodel must be None or direct".format(self.model))

            self.Z1 = self.Zeta1[self.k_star1, :]  # k_starのZの座標N*L(L=2
            self.Z2 = self.Zeta2[self.k_star2, :]  # k_starのZの座標N*L(L=2

            self.history['y'][epoch, :, :] = self.Y
            self.history['z1'][epoch, :] = self.Z1
            self.history['z2'][epoch, :] = self.Z2
            self.history['self.sigma1'][epoch] = self.sigma1
            self.history['self.sigma2'][epoch] = self.sigma2