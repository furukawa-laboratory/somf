import numpy as np
from libs.tools.create_zeta import create_zeta
from libs.datasets.real.beverage import load_data
from scipy.spatial import distance as dist

#データのimport
data_set  =load_data(ret_situation_label=False,ret_beverage_label=False)
X=data_set[0]

N1=X.shape[0]
N2=X.shape[1]
N3=X.shape[2]
K1=5
K2=10
K3=15

#近傍半径の計算
sigma1_max=1.0
sigma1_min=0.1
sigma2_max=1.0
sigma2_min=0.1
sigma3_max=1.0
sigma3_min=0.1
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

#モード1の学習量の計算
sigma1=max(sigma1_min,sigma1_max*(-1/tau1))
Dist_zeta1=dist.cdist(Z1,Zeta1,'sqeuclidean')#N1*K1
H1=np.exp(-1/(2*sigma1*sigma1)*Dist_zeta1)
G1=np.sum(H1,axis=0)
G1inv = np.reciprocal(G1) # Gのそれぞれの要素の逆数を取る
R1=H1*G1inv

sigma2=max(sigma2_min,sigma2_max*(-1/tau2))
Dist_zeta2=dist.cdist(Z2,Zeta2,'sqeuclidean')#N2*K2
H2=np.exp(-1/(2*sigma2*sigma2)*Dist_zeta2)
G2=np.sum(H2,axis=0)
G2inv = np.reciprocal(G2) # Gのそれぞれの要素の逆数を取る
R2=H2*G2inv


sigma3=max(sigma3_min,sigma3_max*(-1/tau3))
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

#これはさすがにeigsum使わない方がいいかも...
#Y=np.einsum('il,jm,kn,ijk->lmn',R1,R2,R3,X)#K1*K2*N3
#print(Y.shape)