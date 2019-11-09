#+型階層TSOMのクラスを追加するプログラム
from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from scipy.spatial import distance as dist

class TSOMPlusSOMWatanabe:
    def __init__(self,member_features,index_members_of_group,params_tsom, params_som):
        self.index_members_of_group = index_members_of_group
        self.params_tsom = params_tsom
        self.params_som = params_som

        self.params_tsom['X'] = member_features

    def fit(self,tsom_epoch_num,kernel_width,som_epoch_num):
        self._fit_1st_TSOM(tsom_epoch_num)
        self._fit_KDE(kernel_width)
        self._fit_2nd_SOM(som_epoch_num)

    def _fit_1st_TSOM(self,tsom_epoch_num):
        self.tsom = TSOM2(**self.params_tsom)
        self.tsom.fit(tsom_epoch_num)

    def _fit_KDE(self,kernel_width):#学習した後の潜在空間からKDEで確率分布を作る
        group_density = np.empty((0,self.tsom.Zeta1.shape[0]))
        for members in self.index_members_of_group:
            Dist = dist.cdist(self.tsom.Zeta1,self.tsom.Z1[members],'sqeuclidean') #KxN
            kernel_value = np.exp(-0.5*Dist/(kernel_width*kernel_width)).sum(axis=1) #K
            density = kernel_value / kernel_value.sum()
            group_density = np.append(group_density, density[None,:], axis=0)

        self.params_som['X'] = group_density
        self.params_som['metric'] = 'KLdivergence'

    def _fit_2nd_SOM(self,som_epoch_num):#上位のSOMを
        self.som = SOM(**self.params_som)
        self.som.fit(som_epoch_num)
