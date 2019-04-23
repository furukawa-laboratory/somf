#+型階層TSOMのクラスを追加するプログラム
from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from scipy.spatial import distance as dist

class TSOM_plus_SOM:
    def __init__(self,input_data,init,group_label,*args):
        #とりあえず、keyは固定にして場所自由でいいかも.一部、tsomの時にtupleになっている場合の処理を追加
        #下位のTSOMのパラメータ設定
        self.tsom_latent_dim=args[0][0]
        self.tsom_resolution = args[1][0]
        self.tsom_sigma_max=args[2][0]
        self.tsom_sigma_min=args[3][0]
        self.tsom_tau=args[4][0]
        #上位のSOMのパラメータ設定
        self.som_latent_dim=args[0][1]
        self.som_resolution = args[1][1]
        self.som_sigma_max = args[2][1]
        self.som_sigma_min = args[3][1]
        self.som_tau = args[4][1]
        Init=init


        self.input_data=input_data#下位のTSOMに入れるパラメータ
        self.group_label = group_label # グループ数の確認
        self.group_num=len(self.group_label)

        #上位のSOMのパラメータ設定と、下位TSOMのパラメータ設定を引数として決めてやる必要がある.
        self.tsom=TSOM2(self.input_data,latent_dim=self.tsom_latent_dim,resolution=self.tsom_resolution,SIGMA_MAX=self.tsom_sigma_max
                        ,SIGMA_MIN=self.tsom_sigma_min,init=Init,TAU=self.tsom_tau)
        self.prob_data = np.zeros((self.group_num, self.tsom.K1))  # group数*ノード数


    def fit_1st_TSOM(self,tsom_epoch_num):
        self.tsom.fit(tsom_epoch_num)

    def fit_KDE(self,kernel_width):#学習した後の潜在空間からKDEで確率分布を作る
        #グループごとにKDEを適用
        for i in range(self.group_num):
            Dist=dist.cdist(self.tsom.Zeta1, self.tsom.Z1[self.group_label[i],:], 'sqeuclidean')# KxNの距離行列を計算
            H = np.exp(-Dist / (2 * kernel_width * kernel_width))  # KxNの学習量行列を計算
            prob = np.sum(H, axis=1)#K*1
            prob_sum = np.sum(prob)#1*1
            prob = prob / prob_sum#K*1
            self.prob_data[i,:]=prob

    def fit_2nd_SOM(self,som_epoch_num,init):#上位のSOMを
        self.som = SOM(self.prob_data, latent_dim=self.som_latent_dim, resolution=self.som_resolution,
                       sigma_max=self.som_sigma_max,sigma_min=self.som_sigma_min, tau=self.som_tau, init=init, metric="KLdivergence")
        self.som.fit(som_epoch_num)

def _main():
    #グループ数分のガウス分布を生成してそれぞれサンプルを生成する.

    group_num=10#group数
    input_dim=3#各メンバーの特徴数
    samples_per_group=30#各グループにメンバーに何人いるのか
    #平均ベクトルを一様分布から生成
    mean=np.random.rand(group_num,input_dim)

    input_data=np.zeros((group_num,samples_per_group,input_dim))#input dataは1stTSOMに入れるデータ

    for i in range(group_num):
        samples=np.random.multivariate_normal(mean=mean[i],cov=np.identity(input_dim),size=samples_per_group)
        input_data[i,:,:]=samples
    input_data = input_data.reshape((group_num * samples_per_group, input_dim))
    group_label=np.zeros((group_num,samples_per_group),dtype=int)
    #init='random'
    Z1 = np.random.rand(group_num*samples_per_group,2) * 2.0 - 1.0
    Z2 = np.random.rand(input_dim, 1) * 2.0 - 1.0
    init=(Z1,Z2)

    # #dictのパラメータ名は固定latent_dim,resolution,sigma_max,sigma_min,tauでSOMとTSOMでまとめる
    htsom=TSOM_plus_SOM(input_data,init,group_label,((2,1),2),((5,10),10),(1.0,1.0),(0.1,0.1),(50,50))

    #plus型TSOM(TSOM*SOM)のやつ
    htsom.fit_1st_TSOM(tsom_epoch_num=250)#1stTSOMの学習
    htsom.fit_KDE(kernel_width=1.0)#カーネル密度推定を使って2ndSOMに渡す確率分布を作成
    htsom.fit_2nd_SOM(som_epoch_num=250)#2ndSOMの学習

if __name__ == '__main__':
    _main()