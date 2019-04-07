#+型階層TSOMのクラスを追加するプログラム
from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np

class TSOM_plus_SOM:
    def __init__(self,input_data,kernel_width,firstTSOM_parameters,secondSOM_parameters):

        self.firstTSOM_parameters=firstTSOM_parameters#辞書でよい？
        self.second_parameters=secondSOM_parameters#辞書でよいか？
        self.input_data=input_data#下位のTSOMに入れるパラメータ
        self.kernel_width=kernel_width#KDEのカーネル幅を決める(ガウスカーネル)

        #上位のSOMのパラメータ設定と、下位TSOMのパラメータ設定を引数として決めてアyる必要がある.
        self.tsom=TSOM2(self.input_data,latent_dim=2,resolution=10,SIGMA_MAX=1.0,SIGMA_MIN=0.1,init='random',TAU=50)
        #self.som=SOM(X,latent_dim=,resolution=,sigma_max=,sigma_min=,tau=,init="random",metric="KLdivergence")
        #いるやつ:TSOMクラス,SOMクラス,KDE関数


    def fit_1st_TSOM(self):
        pass

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