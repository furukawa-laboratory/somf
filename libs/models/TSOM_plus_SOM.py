#+型階層TSOMのクラスを追加するプログラム
from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from scipy.spatial import distance as dist

class TSOM_plus_SOM:
    def __init__(self,input_data,group_label,**kargs):
        #とりあえず、keyは固定にして場所自由でいいかも.一部、tsomの時にtupleになっている場合の処理を追加
        #下位のTSOMのパラメータ設定
        self.tsom_latent_dim=kargs['tsom_latentdim']
        self.tsom_resolution = kargs['tsom_resolution']
        self.tsom_sigma_max=kargs['tsom_sigma_max']
        self.tsom_sigma_min=kargs['tsom_sigma_min']
        self.tsom_tau=kargs['tsom_tau']

        #上位のSOMのパラメータ設定
        self.som_latent_dim=kargs['som_latentdim']
        self.som_sigma_max = kargs['som_sigma_max']
        self.som_sigma_min = kargs['som_sigma_min']
        self.som_tau = kargs['som_tau']
        self.som_resolution = kargs['som_resolution']

        self.input_data=input_data#下位のTSOMに入れるパラメータ
        self.group_label = group_label # グループ数の確認
        self.group_num=len(self.group_label)

        #上位のSOMのパラメータ設定と、下位TSOMのパラメータ設定を引数として決めてアyる必要がある.
        self.tsom=TSOM2(self.input_data,latent_dim=self.tsom_latent_dim,resolution=self.tsom_resolution,SIGMA_MAX=self.tsom_sigma_max
                        ,SIGMA_MIN=self.tsom_sigma_min,init='random',TAU=self.tsom_tau)
        self.output_data = np.zeros((self.group_num, self.tsom.K1))  # group数*ノード数


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
            self.output_data[i,:]=prob

    def fit_2nd_SOM(self,som_epoch_num):#上位のSOMを
        self.som = SOM(self.output_data, latent_dim=self.som_latent_dim, resolution=self.som_resolution,
                       sigma_max=self.som_sigma_max,sigma_min=self.som_sigma_min, tau=self.som_tau, init="random", metric="KLdivergence")
        self.som.fit(som_epoch_num)

#もやもや:1stと2ndのパラメータは辞書で入れてもらう？(可変長引数でやろう!)
def _main():
    #3つのガウス分布でそれぞれサンプルを生成して,その情報をグループ情報とする
    group1_mean=np.array([0.0,0.0,0.0])
    group1_cov=np.identity(n=3)
    group2_mean = np.array([-1.0, -1.0, -1.0])
    group2_cov = np.identity(n=3)
    group3_mean = np.array([1.0, 1.0, 1.0])
    group3_cov = np.identity(n=3)

    group1_data=np.random.multivariate_normal(mean=group1_mean,cov=group1_cov,size=20)
    group2_data = np.random.multivariate_normal(mean=group2_mean, cov=group2_cov,size=20)
    group3_data = np.random.multivariate_normal(mean=group3_mean, cov=group3_cov, size=20)
    input_data=np.concatenate((group1_data,group2_data,group3_data),axis=0)

    group1_label=np.arange(group1_data.shape[0])
    group2_label = np.arange(group2_data.shape[0])+group1_data.shape[0]
    group3_label=np.arange(group3_data.shape[0])+group1_data.shape[0]+group2_data.shape[0]
    group_label=(group1_label,group2_label,group3_label)

    #dictのパラメータ名は固定
    htsom=TSOM_plus_SOM(input_data=input_data,group_label=group_label,tsom_resolution=10,tsom_latentdim=2 ,tsom_sigma_max=1.0,
                        tsom_sigma_min=0.1,tsom_tau=50,som_latentdim=2,som_resolution=10,som_sigma_max=1.0,som_sigma_min=0.1,som_tau=50)
    #htsom内で呼び出しているtsomのクラス内の変数を参照できるか？→selfつければできるよ！
    htsom.fit_1st_TSOM(tsom_epoch_num=250)
    htsom.fit_KDE(kernel_width=1.0)
    htsom.fit_2nd_SOM(som_epoch_num=250)

if __name__ == '__main__':
    _main()