import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from datasets.kura_tsom import load_kura_tsom
from tqdm import tqdm
import ComponentPlane as comp
#from libs.datasets.artificial.kura import create_data
#from libs.datasets.artificial.animal import load_data
#from libs.visualization.som.animation_learning_process_3d import anime_learning_process_3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#データの生成
I=10
J=20
X=load_kura_tsom(I,J)
nodes1_kx=20
nodes1_ky=1
nodes1_num=nodes1_kx*nodes1_ky
nodes2_kx=20
nodes2_ky=1
nodes2_num=nodes2_kx*nodes2_ky
mode1_samples=X.shape[0]
mode2_samples=X.shape[1]
observed_dim=X.shape[2]
tau1=50
tau2=50
sigma1_min=0.1
sigma1_zero=1.0
sigma2_min=0.1
sigma2_zero=1.0
epoch_num=250

#モデルの初期化
Y=np.random.rand(nodes1_num,nodes2_num,observed_dim)*2.0-1.0#参照ベクトルの初期化0~1 Y:K*D
U=np.random.rand(mode1_samples,nodes2_num,observed_dim)*2.0-1.0
V=np.random.rand(nodes1_num,mode2_samples,observed_dim)*2.0-1.0
# '潜在空間の生成'
mode1_x = np.linspace(-1, 1, nodes1_kx)
mode1_y = np.linspace(-1, 1, nodes1_ky)
mode2_x = np.linspace(-1, 1, nodes2_kx)
mode2_y = np.linspace(-1, 1, nodes2_ky)
mode1_Zeta1, mode1_Zeta2 = np.meshgrid(mode1_x, mode1_y)
mode2_Zeta1, mode2_Zeta2 = np.meshgrid(mode2_x, mode2_y)
Zeta1 = np.c_[mode1_Zeta2.ravel(), mode1_Zeta1.ravel()]
Zeta2 = np.c_[mode2_Zeta2.ravel(), mode2_Zeta1.ravel()]
Y_allepoch=np.zeros((epoch_num,nodes1_num,nodes2_num,observed_dim))
#memo: 組ませるのであれば，アルゴリズムのイメージを持ってもらわないと厳しい?
mode1_D=np.zeros((I,nodes1_num))
mode2_D=np.zeros((J,nodes2_num))

for epoch in np.arange(epoch_num):
    print(epoch)
    #競合過程を作る
    #mode1の競合過程
    for i in np.arange(I):
        for k in np.arange(nodes1_num):
            distance2=0
            for l in np.arange(nodes2_num):
                distance=0
                for d in np.arange(observed_dim):
                    distance+=(U[i][l][d]-Y[k][l][d])**2
                distance2+=distance
            mode1_D[i][k]=distance2

    k_star=np.argmin(mode1_D,axis=1)

    #mode2の競合過程
    for j in np.arange(J):
        for l in np.arange(nodes2_num):
            distance2 = 0
            for k in np.arange(nodes1_num):
                distance=0
                for d in np.arange(observed_dim):
                    distance+=(V[k][j][d]-Y[k][l][d])**2
                distance2+=distance
            mode2_D[j][l]=distance2

    l_star=np.argmin(mode2_D,axis=1)

    #協調過程

    h1=np.zeros((I,nodes1_num))
    h2=np.zeros((J,nodes2_num))

    #mode1の学習量の計算
    sigmak= sigma1_min + (sigma1_zero - sigma1_min) * np.exp(-epoch / tau1)
    for i in np.arange(I):
        for k in np.arange(nodes1_num):
            zeta_dis1=0
            for latent_l in np.arange(Zeta1.shape[1]):
                zeta_dis1+=(Zeta1[k_star[i]][latent_l]-Zeta1[k][latent_l])**2
            h1[i][k]=np.exp(-0.5*(zeta_dis1*zeta_dis1)/sigmak**2)


    #mode2の学習量の計算
    sigmal = sigma2_min + (sigma2_zero - sigma2_min) * np.exp(-epoch / tau2)
    for j in np.arange(J):
        for l in np.arange(nodes2_num):
            zeta_dis2=0
            for latent_l in np.arange(Zeta2.shape[1]):
                zeta_dis2+=(Zeta2[l_star[j]][latent_l]-Zeta2[l][latent_l])**2
            h2[j][l]=np.exp(-0.5*(zeta_dis2*zeta_dis2)/sigmal**2)

    #適応過程の計算
    #gの計算
    #mode1のgの計算
    for k in np.arange(nodes1_num):
        g1=0
        for i in np.arange(I):
            g1+=h1[i][k]
        for i in np.arange(I):
            h1[i][k] /=g1

    #mode2のgの計算
    for l in np.arange(nodes2_num):
        g2=0
        for j in np.arange(J):
            g2+=h2[j][l]
        for j in np.arange(J):
            h2[j][l] /=g2

    #モデルの更新
    #1次モデル
    U = np.zeros((I,nodes2_num,observed_dim))
    V = np.zeros((nodes1_num,J, observed_dim))
    Y = np.zeros((nodes1_num, nodes2_num, observed_dim))
    for i in np.arange(I):
        for l in np.arange(nodes2_num):
            for d in np.arange(observed_dim):
                for j in np.arange(J):
                    U[i][l][d]+=h2[j][l]*X[i][j][d]

    for k in np.arange(nodes1_num):
        for j in np.arange(J):
            for d in np.arange(observed_dim):
                for i in np.arange(I):
                    V[k][j][d]+=h1[i][k]*X[i][j][d]

    #2次モデルの更新
    for k in np.arange(nodes1_num):
        for l in np.arange(nodes2_num):
            for d in np.arange(observed_dim):
                for i in np.arange(I):
                    for j in np.arange(J):
                        Y[k][l][d]+=h1[i][k]*h2[j][l]*X[i][j][d]
    Y_allepoch[epoch,:,:]=Y


fig = plt.figure()
ax = Axes3D(fig)
def plot(i):
    ax.cla()
    ax.scatter(X[:,:, 0], X[:,:, 1], X[:,:, 2])
    ax.plot_wireframe(Y_allepoch[i,:, :, 0], Y_allepoch[i,:, :, 1], Y_allepoch[i,:, :, 2])
    plt.title(' t=' + str(i))

ani = animation.FuncAnimation(fig, plot, frames=epoch_num,interval=10)
plt.show()