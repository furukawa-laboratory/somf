#+型階層TSOMのクラスを追加するプログラム
from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np

class TSOM_plus_SOM:
    def __init__(self,input_data,kernel_width,tsom_latentdim,tsom_resolution,tsom_sigma_max,tsom_sigma_min,tsom_tau,tsom_epoch_num,
                 som_latentdim,som_resolution,som_sigma_max,,som_sigma_min,som_tau):

        #下位のTSOMのパラメータ設定(とりあえず、全モード共通)
        self.tsom_latent_dim=tsom_latentdim
        self.tsom_sigma_max=tsom_sigma_max
        self.tsom_sigma_min=tsom_sigma_min
        self.tsom_tau=tsom_tau
        self.tsom_resolution=tsom_resolution
        self.tsom_epoch_num = tsom_epoch_num
        #上位のSOMのパラメータ設定
        self.som_latent_dim=som_latentdim
        self.som_sigma_max = som_sigma_max
        self.som_sigma_min = som_sigma_min
        self.som_tau = som_tau
        self.som_resolution = som_resolution



        self.input_data=input_data#下位のTSOMに入れるパラメータ
        self.kernel_width=kernel_width#KDEのカーネル幅を決める(ガウスカーネル)

        #上位のSOMのパラメータ設定と、下位TSOMのパラメータ設定を引数として決めてアyる必要がある.
        self.tsom=TSOM2(self.input_data,latent_dim=self.tsom_latent_dim,resolution=self.tsom_resolution,SIGMA_MAX=self.tsom_sigma_max
                        ,SIGMA_MIN=self.tsom_sigma_min,init='random',TAU=self.tsom_tau)
        #self.som=SOM(,latent_dim=,resolution=,sigma_max=,sigma_min=,tau=,init="random",metric="KLdivergence")
        #いるやつ:TSOMクラス,SOMクラス,KDE関数


    def fit_1st_TSOM(self):
        self.tsom.fit(self.tsom_epoch_num)

    def fit_KDE(self):#学習した後の潜在空間からKDEで確率分布を作る
        pass

    def fit_2nd_SOM(self):#上位のSOMを
        pass

#もやもや:1stと2ndのパラメータは辞書で入れてもらう？
def _main():
    N=10
    D=20
    input_data=np.random.rand(N,D)
    firstTSOM_parameters=1.0
    secondSOM_parameters=1.0
    htsom=TSOM_plus_SOM(input_data,kernel_width=1.0,firstTSOM_parameters=firstTSOM_parameters,secondSOM_parameters=secondSOM_parameters)
    #htsom内で呼び出しているtsomのクラス内の変数を参照できるか？→selfつければできるよ！

if __name__ == '__main__':
    _main()