#+型階層TSOMのクラスを追加するプログラム
from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from scipy.spatial import distance as dist

class TSOM_plus_SOM:
    def __init__(self,input_data,tsom_latentdim,tsom_resolution,tsom_sigma_max,tsom_sigma_min,tsom_tau,tsom_epoch_num,
                 som_latentdim,som_resolution,som_sigma_max,som_sigma_min,som_tau):

        #下位のTSOMのパラメータ設定(とりあえず、全モード共通)
        self.tsom_latent_dim=tsom_latentdim
        self.tsom_sigma_max=tsom_sigma_max
        self.tsom_sigma_min=tsom_sigma_min
        self.tsom_tau=tsom_tau
        self.tsom_resolution=tsom_resolution
        #上位のSOMのパラメータ設定
        self.som_latent_dim=som_latentdim
        self.som_sigma_max = som_sigma_max
        self.som_sigma_min = som_sigma_min
        self.som_tau = som_tau
        self.som_resolution = som_resolution



        self.input_data=input_data#下位のTSOMに入れるパラメータ
        #self.kernel_width=kernel_width#KDEのカーネル幅を決める(ガウスカーネル)

        #上位のSOMのパラメータ設定と、下位TSOMのパラメータ設定を引数として決めてアyる必要がある.
        self.tsom=TSOM2(self.input_data,latent_dim=self.tsom_latent_dim,resolution=self.tsom_resolution,SIGMA_MAX=self.tsom_sigma_max
                        ,SIGMA_MIN=self.tsom_sigma_min,init='random',TAU=self.tsom_tau)
        #self.som=SOM(,latent_dim=,resolution=,sigma_max=,sigma_min=,tau=,init="random",metric="KLdivergence")
        #いるやつ:TSOMクラス,SOMクラス,KDE関数


    def fit_1st_TSOM(self,tsom_epoch_num):
        self.tsom.fit(tsom_epoch_num)

    def fit_KDE(self,kernel_width,group_label):#学習した後の潜在空間からKDEで確率分布を作る
        group_num=len(group_label)#グループ数の確認
        group_1_Z=self.tsom.Z1[group_label[0]]
        group_2_Z = self.tsom.Z1[group_label[1]]
        group_3_Z = self.tsom.Z1[group_label[2]]

        Dist1 = dist.cdist(self.tsom.Zeta1, group_1_Z, 'sqeuclidean')
        # KxNの距離行列を計算
        # ノードと勝者ノードの全ての組み合わせにおける距離を網羅した行列
        H1 = np.exp(-Dist1 / (2 * kernel_width * kernel_width))  # KxNの学習量行列を計算
        prob1=np.sum(H1,axis=1)
        prob1_sum=np.sum(prob1)
        prob1=prob1/prob1_sum

        Dist2 = dist.cdist(self.tsom.Zeta1, group_2_Z, 'sqeuclidean')
        H2 = np.exp(-Dist2 / (2 * kernel_width * kernel_width))  # KxNの学習量行列を計算
        prob2 = np.sum(H2, axis=1)
        prob2_sum = np.sum(prob2)
        prob2 = prob2 / prob2_sum

        Dist3 = dist.cdist(self.tsom.Zeta1, group_3_Z, 'sqeuclidean')
        H3 = np.exp(-Dist3 / (2 * kernel_width * kernel_width))  # KxNの学習量行列を計算
        prob3 = np.sum(H3, axis=1)
        prob3_sum = np.sum(prob3)
        prob3 = prob3 / prob3_sum

        output_data=np.concatenate((prob1[:,np.newaxis],prob2[:,np.newaxis],prob3[:,np.newaxis]),axis=1)
        self.output_data=output_data.T
        

    def fit_2nd_SOM(self):#上位のSOMを
        pass

#もやもや:1stと2ndのパラメータは辞書で入れてもらう？
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
    print(group1_label)
    print(group2_label)
    print(group3_label)
    group_label=(group1_label,group2_label,group3_label)

    tsom_latentdim=2
    tsom_resolution=10
    tsom_sigma_max=1.0
    tsom_sigma_min=0.1
    tsom_tau=50
    tsom_epoch_num=250
    som_latentdim=2
    som_resolution=10
    som_sigma_max=1.0
    som_sigma_min=0.1
    som_tau=50

    htsom=TSOM_plus_SOM(input_data,tsom_latentdim,tsom_resolution,tsom_sigma_max,tsom_sigma_min,tsom_tau,tsom_epoch_num,
                        som_latentdim,som_resolution,som_sigma_max,som_sigma_min,som_tau)
    #htsom内で呼び出しているtsomのクラス内の変数を参照できるか？→selfつければできるよ！
    htsom.fit_1st_TSOM(tsom_epoch_num=250)
    htsom.fit_KDE(kernel_width=1.0,group_label=group_label)

if __name__ == '__main__':
    _main()