from libs.models.tsom import TSOM2
from libs.models.som import SOM
import numpy as np
from tqdm import tqdm


class TSOMCrossSOM:
    def __init__(self, datasets, latent_dim,resolution,SIGMA_MAX, SIGMA_MIN, TAU,init):

        # #データセットに関しての例外処理(tupleかlistのみでndarrayの場合はerror)
        if isinstance(datasets,(tuple,list)):
            self.datasets=datasets
            self.class_num = len(datasets)  # クラス数
        else:
            raise ValueError("invalid datasets: {}\ndataset must be list or tuple".format(datasets))

        #全クラスで次元数が違う場合でエラーを出すかどうか？

        #resolutionに関しての例外処理(下位のTSOMとSOMで分けてる)
        if isinstance(resolution,(list,tuple)) and len(resolution)==2:
            self.child_resolution=resolution[0]
            self.parent_resolution = resolution[1]
        elif isinstance(resolution,int):
            self.child_resolution=resolution
            self.parent_resolution = resolution
        else:
            raise ValueError("invalid resolution: {}".format(resolution))

        #sigma_maxについての例外処理(下位のTSOMとSOMで分けてる)
        if isinstance(SIGMA_MAX,(tuple,list)):
            self.child_sigma_max = SIGMA_MAX[0]
            self.parent_sigma_max = SIGMA_MAX[1]
        elif isinstance(SIGMA_MAX,float):#下位のTSOMと上位のSOMで同じ場合
            self.child_sigma_max=SIGMA_MAX
            self.parent_sigma_max = SIGMA_MAX
        else:
            raise ValueError("invalid SIGMAX: {}".format(SIGMA_MAX))

        # sigma_minについての例外処理(下位のTSOMとSOMで分けてる)
        if isinstance(SIGMA_MIN, (tuple, list)):
            self.child_sigma_min = SIGMA_MIN[0]
            self.parent_sigma_min = SIGMA_MIN[1]
        elif isinstance(SIGMA_MIN, float):  # 下位のTSOMと上位のSOMで同じ場合
            self.child_sigma_min = SIGMA_MIN
            self.parent_sigma_min = SIGMA_MIN
        else:
            raise ValueError("invalid SIGMIN: {}".format(SIGMA_MIN))

        #TAUについての例外処理(下位のTSOMとSOMで分けてる)
        if isinstance(TAU,(tuple,list)):
            self.child_tau=TAU[0]
            self.parent_tau=TAU[1]
        elif isinstance(TAU,int):
            self.child_tau = TAU
            self.parent_tau = TAU
        else:
            raise ValueError("invalid TAU: {}".format(TAU))

        #latent_dimについての例外処理(下位のTSOMとSOMで分けてる)
        if isinstance(latent_dim,(tuple,list)):
            self.child_latent_dim = latent_dim[0]
            self.parent_latent_dim = latent_dim[1]
        elif isinstance(latent_dim,int):
            self.child_latent_dim=latent_dim
            self.parent_latent_dim = latent_dim
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))

        #initについての例外処理(上位と下位に分けてる)
        if isinstance(init,(tuple,list)):
            self.child_init=init[0]
            self.parent_init = init[1]
        else:
            raise ValueError("invalid init: {}\n init must be list or tuple".format((init)))


    def fit(self, epoch_num=100):

        # 下位TSOMの定義
        child_TSOM = []
        for i in np.arange(self.class_num):
            temp_class = TSOM2(X=self.datasets[i], latent_dim=self.child_latent_dim, resolution=self.child_resolution,
                               SIGMA_MAX=self.child_sigma_max, SIGMA_MIN=self.child_sigma_min, TAU=self.child_tau)
            child_TSOM.append(temp_class)


        parent_observed_dim = child_TSOM[0].K1 * child_TSOM[0].K2 * child_TSOM[0].observed_dim
        reference_vector_set = np.zeros((class_num, parent_observed_dim))

        for epoch in np.arange(epoch_num):

            #学習が1回目の時(潜在変数初期化なので別にする)
            if epoch==0:
                # childTSOMの学習(1回目)
                for i in np.arange(self.class_num):
                    child_TSOM[i].fit(nb_epoch=1)

                # parentに渡すデータの作成
                for i in np.arange(self.class_num):
                    reshaped_reference_vector = child_TSOM[i].Y.reshape((parent_observed_dim))
                    reference_vector_set[i, :] = reshaped_reference_vector

                # 上位SOMの定義
                parent_SOM = SOM(X=reference_vector_set, latent_dim=self.parent_latent_dim,
                                 resolution=self.parent_resolution
                                 , sigma_max=self.parent_sigma_max, sigma_min=self.parent_sigma_min,
                                 tau=self.parent_tau,
                                 init=self.parent_init)
                # 上位のSOMの学習(１回目)
                parent_SOM.fit(nb_epoch=1)

                # コピーバック
                for i in np.arange(self.class_num):
                    child_TSOM[i].Y = parent_SOM.Y[parent_SOM.bmus[i]]

            else:
                #下位のTSOMの学習
                for i in np.arange(self.class_num):
                    # 勝者決定
                    child_TSOM[i]._competitive_process_nonmissing_indirect()

                    #学習量の決定
                    child_TSOM[i]._cooperative_process(epoch=epoch)

                    #写像の更新
                    child_TSOM[i]._adaptive_process_nonmissing_indirect()

                    #parentに渡すデータん更新
                    reshaped_reference_vector = child_TSOM[i].Y.reshape((parent_observed_dim))
                    reference_vector_set[i, :] = reshaped_reference_vector

                #上位のSOMの学習
                parent_SOM._competitive_process()
                parent_SOM._cooperative_process(epoch=epoch)
                parent_SOM._adaptive_process()

                #コピーバック
                for i in np.arange(self.class_num):
                    child_TSOM[i].Y = parent_SOM.Y[parent_SOM.bmus[i]]






if __name__ == '__main__':

    #人工データの作成
    class_num=10
    N1=20#ひとまず全クラスで同じ状況で行う
    N2=30
    observed_dim=4
    dataset=[]
    for i in np.arange(class_num):
        Xi=np.random.rand(N1,N2,observed_dim)
        dataset.append(Xi)
    #パラメータの設定
    child_init="random"
    parent_init = "random"
    init=[child_init,parent_init]

    tsom_cross_som=TSOMCrossSOM(datasets=dataset,latent_dim=2,resolution=20,SIGMA_MAX=1.0,SIGMA_MIN=0.1,TAU=40,init=init)

    tsom_cross_som.fit(epoch_num=1)
