import numpy as np
from libs.tools.create_zeta import create_zeta
from libs.datasets.real.beverage import load_data

#データのimport
data_set  =load_data(ret_situation_label=False,ret_beverage_label=False)
X=data_set[0]

N1=X.shape[0]
N2=X.shape[1]
N3=X.shape[2]
K1=5
K2=10
K3=15

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

#学習量の作成
H1=np.zeros((N1,K1))
H2=np.zeros((N2,K2))
H3=np.zeros((N3,K3))


#アルゴリズム
#モード1の勝者決定
Dist=U1[:,np.newaxis,:,:]-Y[np.newaxis,:,:,:]#N1*K1*K2*K3
Dist_sum=np.sum(Dist,axis=(2,3))#N1*K1
k1_star=np.argmin(Dist_sum,axis=1)#N1*1

#モード2の勝者決定
Dist2=U2[:,:,np.newaxis,:]-Y[:,np.newaxis,:,:]#K1*N2*K2*K3
Dist2_sum=np.sum(Dist2,axis=(0,2))#N2*K2
k2_star=np.argmin(Dist2_sum,axis=1)

#モード3の勝者決定
Dist3=U3[:,:,:,np.newaxis]-Y[:,:,np.newaxis,:]#K1*K2*N3*K3
Dist3_sum=np.sum(Dist3,axis=(0,1))#N3*K3
k3_star=np.argmin(Dist3_sum,axis=1)#N3*1



