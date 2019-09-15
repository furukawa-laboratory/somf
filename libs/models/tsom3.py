import numpy as np
from libs.tools.create_zeta import create_zeta
from libs.datasets.real.beverage import load_data
from scipy.spatial import distance as dist
from tqdm import tqdm
import matplotlib.pyplot as plt

#データのimport
data_set  =load_data(ret_situation_label=True,ret_beverage_label=True)
X=data_set[0]
beverage_label=data_set[1]
situation_label=data_set[2]

N1=X.shape[0]
N2=X.shape[1]
N3=X.shape[2]
K1=5
K2=5
K3=5
epoch_num=250

#近傍半径の計算
sigma1_max=0.1
sigma1_min=0.01
sigma2_max=1.5
sigma2_min=0.3
sigma3_max=1.5
sigma3_min=0.3
tau1=50
tau2=50
tau3=50


#潜在空間の作成
Zeta1=create_zeta(zeta_min=-1, zeta_max=1, latent_dim=2, resolution=K1, include_min_max=True)
Zeta2=create_zeta(zeta_min=-1, zeta_max=1, latent_dim=2, resolution=K2, include_min_max=True)
Zeta3=create_zeta(zeta_min=-1, zeta_max=1, latent_dim=2, resolution=K3, include_min_max=True)

#勝者ユニットの初期化
Z1=np.random.rand(N1,2)*2-1
Z2=np.random.rand(N2,2)*2-1
Z3=np.random.rand(N3,2)*2-1


#参照ベクトルの作成
Y=np.zeros((K1,K2,K3))
U1=np.zeros((N1,K2,K3))
U2=np.zeros((K1,N2,K3))
U3=np.zeros((K1,K2,N3))



for epoch in tqdm(range(epoch_num)):
    #モード1の学習量の計算
    sigma1=max(sigma1_min,sigma1_max*(-epoch/tau1))
    Dist_zeta1=dist.cdist(Z1,Zeta1,'sqeuclidean')#N1*K1
    H1=np.exp(-1/(2*sigma1*sigma1)*Dist_zeta1)
    G1=np.sum(H1,axis=0)
    G1inv = np.reciprocal(G1) # Gのそれぞれの要素の逆数を取る
    R1=H1*G1inv

    sigma2=max(sigma2_min,sigma2_max*(-epoch/tau2))
    Dist_zeta2=dist.cdist(Z2,Zeta2,'sqeuclidean')#N2*K2
    H2=np.exp(-1/(2*sigma2*sigma2)*Dist_zeta2)
    G2=np.sum(H2,axis=0)
    G2inv = np.reciprocal(G2) # Gのそれぞれの要素の逆数を取る
    R2=H2*G2inv


    sigma3=max(sigma3_min,sigma3_max*(-epoch/tau3))
    Dist_zeta3=dist.cdist(Z3,Zeta3,'sqeuclidean')#N3*K3
    H3=np.exp(-1/(2*sigma3*sigma3)*Dist_zeta3)
    G3=np.sum(H3,axis=0)
    G3inv = np.reciprocal(G3) # Gのそれぞれの要素の逆数を取る
    R3=H3*G3inv

    #写像の計算
    #1次モデルの計算
    #データ: i,j,k
    #ノード: l,m,n
    U1= np.einsum('jm,kn,ijk->imn',R2,R3,X)#N1*K1*K2
    U2=np.einsum('il,kn,ijk->ljn',R1,R3,X)#K1*N2*K3
    U3=np.einsum('il,jm,ijk->lmk',R1,R2,X)#K1*K2*N3

    #一時モデルを使って2次モデルを更新
    Y=np.einsum('il,imn->lmn',R1,U1)#K1*K2*K3

    # アルゴリズム
    # モード1の勝者決定
    Dist = U1[:, np.newaxis, :, :] - Y[np.newaxis, :, :, :]  # N1*K1*K2*K3
    Dist_sum = np.sum(Dist, axis=(2, 3))  # N1*K1
    k1_star = np.argmin(Dist_sum, axis=1)  # N1*1

    # モード2の勝者決定
    Dist2 = U2[:, :, np.newaxis, :] - Y[:, np.newaxis, :, :]  # K1*N2*K2*K3
    Dist2_sum = np.sum(Dist2, axis=(0, 2))  # N2*K2
    k2_star = np.argmin(Dist2_sum, axis=1)

    # モード3の勝者決定
    Dist3 = U3[:, :, :, np.newaxis] - Y[:, :, np.newaxis, :]  # K1*K2*N3*K3
    Dist3_sum = np.sum(Dist3, axis=(0, 1))  # N3*K3
    k3_star = np.argmin(Dist3_sum, axis=1)  # N3*1


fig=plt.figure()
ax=fig.add_subplot(1,2,1)
ax.scatter(Z2[:,0],Z2[:,1])
for i in range(X.shape[1]):
    ax.text(Z2[i,0],Z2[i,1],beverage_label[i])

ax2=fig.add_subplot(1,2,2)
ax2.scatter(Z3[:,0],Z3[:,1])
for i in range(X.shape[2]):
    ax2.text(Z3[i,0],Z3[i,1],situation_label[i])
plt.show()

class TSOM3():
    def __init__(self, X, latent_dim, resolution, SIGMA_MAX, SIGMA_MIN, TAU, init='random'):

        # 入力データXについて
        if X.ndim == 3:
            self.X = X
            self.N1 = self.X.shape[0]
            self.N2 = self.X.shape[1]
            self.N3=self.X.shape[2]
            self.observed_dim = self.X.shape[2]  # 観測空間の次元
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
            self.SIGMA3_MIN= SIGMA_MIN
        elif isinstance(SIGMA_MIN, (list, tuple)):
            self.SIGMA1_MIN = SIGMA_MIN[0]
            self.SIGMA2_MIN = SIGMA_MIN[1]
            self.SIGMA3_MIN=SIGMA_MIN[2]
        else:
            raise ValueError("invalid SIGMA_MIN: {}".format(SIGMA_MIN))

        # 時定数(TAU)の設定
        if type(TAU) is int:
            self.TAU1 = TAU
            self.TAU2 = TAU
            self.TAU3=TAU
        elif isinstance(TAU, (list, tuple)):
            self.TAU1 = TAU[0]
            self.TAU2 = TAU[1]
            self.TAU3=TAU[2]
        else:
            raise ValueError("invalid TAU: {}".format(TAU))

        # resolutionの設定
        if type(resolution) is int:
            resolution1 = resolution
            resolution2 = resolution
            resolution3=resolution
        elif isinstance(resolution, (list, tuple)):
            resolution1 = resolution[0]
            resolution2 = resolution[1]
            resolution3=resolution[2]
        else:
            raise ValueError("invalid resolution: {}".format(resolution))

        # 潜在空間の設定
        if type(latent_dim) is int:  # latent_dimがintであればどちらのモードも潜在空間の次元は同じ
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim
            self.latent_dim3=latent_dim

        elif isinstance(latent_dim, (list, tuple)):
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]
            self.latent_dim3=latent_dim[2]
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))
            # latent_dimがlist,float,3次元以上はエラーかな?
        self.Zeta1 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim1, resolution=resolution1, include_min_max=True)
        self.Zeta2 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim2, resolution=resolution2, include_min_max=True)
        self.Zeta3 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim3, resolution=resolution3, include_min_max=True)

        # K1とK2は潜在空間の設定が終わった後がいいよね
        self.K1 = self.Zeta1.shape[0]
        self.K2 = self.Zeta2.shape[0]
        self.K3=self.Zeta3.shape[0]

        # 勝者ノードの初期化
        self.Z1 = None
        self.Z2 = None
        self.Z3=None
        if isinstance(init, str) and init in 'random':
            self.Z1 = np.random.rand(self.N1, self.latent_dim1) * 2.0 - 1.0
            self.Z2 = np.random.rand(self.N2, self.latent_dim2) * 2.0 - 1.0
            self.Z3 = np.random.rand(self.N3, self.latent_dim3) * 2.0 - 1.0
        elif isinstance(init, (tuple, list)) and len(init) == 2:
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