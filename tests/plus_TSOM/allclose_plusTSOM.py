import unittest

import numpy as np

from libs.models.TSOM_plus_SOM import TSOM_plus_SOM
from tests.plus_TSOM.plus_TSOM_someone import TSOM_plus_SOM_someone

class TestSOM(unittest.TestCase):
    def test_plusTSOM_ishida_vs_test_plusTSOM_someone(self):
        group_num = 10  # group数
        input_dim = 3  # 各メンバーの特徴数
        samples_per_group = 30  # 各グループにメンバーに何人いるのか
        seed = 100
        np.random.seed(seed)
        #初期値を合わせる
        Z1_init=np.random.rand((group_num * samples_per_group,input_dim))
        Z2_init = np.random.rand((group_num * samples_per_group, input_dim))

        #学習データの作成(平均のみが違うガウス分布からサンプリングした人工データを用いる.ガウス分布数がグループ数で,サンプル数がメンバーの数)
        # 平均ベクトルを一様分布から生成
        mean = np.random.rand(group_num, input_dim)

        input_data = np.zeros((group_num, samples_per_group, input_dim))#input dataは1stTSOMに入れるデータ

        #データの生成
        for i in range(group_num):
            samples = np.random.multivariate_normal(mean=mean[i], cov=np.identity(input_dim), size=samples_per_group)
            input_data[i, :, :] = samples

        input_data = input_data.reshape((group_num * samples_per_group, input_dim))
        #グループラベルを作成
        group_label = np.zeros((group_num, samples_per_group), dtype=int)




        # #plus型TSOMの学習
        #dictのパラメータ名は固定latent_dim,resolution,sigma_max,sigma_min,tauでSOMとTSOMでまとめる
        htsom_ishida = TSOM_plus_SOM(input_data, group_label, ((2, 1), 2), ((5, 10), 10), (1.0, 1.0), (0.1, 0.1), (50, 50))
        htsom_someone=TSOM_plus_SOM_someone(input_data, group_label, ((2, 1), 2), ((5, 10), 10), (1.0, 1.0), (0.1, 0.1), (50, 50))


        # plus型TSOM(TSOM*SOM)のやつ
        htsom_ishida.fit_1st_TSOM(tsom_epoch_num=250)  # 1stTSOMの学習
        htsom_someone.fit_1st_TSOM(tsom_epoch_num=250)  # 1stTSOMの学習

        print("allclose_1stTSOM")
        np.testing.assert_allclose(htsom_ishida.tsom.history['y'], htsom_someone.tsom.history['y'])
        np.testing.assert_allclose(htsom_ishida.tsom.history['z1'], htsom_someone.tsom.history['z1'])
        np.testing.assert_allclose(htsom_ishida.tsom.history['z2'], htsom_someone.tsom.history['z2'])


        # htsom_ishida.fit_KDE(kernel_width=1.0)  # カーネル密度推定を使って2ndSOMに渡す確率分布を作成
        # htsom_someone.fit_KDE(kernel_width=1.0)  # カーネル密度推定を使って2ndSOMに渡す確率分布を作成
        # #print("allclose_Kernel_Dnsity_Estimation")
        # #np.testing.assert_allclose(htsom_ishida.prob_data, htsom_someone.prob_data)
        #
        # htsom_ishida.fit_2nd_SOM(som_epoch_num=250)  # 2ndSOMの学習
        # htsom_someone.fit_2nd_SOM(som_epoch_num=250)  # 2ndSOMの学習

       # print("allclose_2ndSOM")
        #np.testing.assert_allclose(htsom_ishida.som.history['y'], htsom_someone.som.history['y'])
        #np.testing.assert_allclose(htsom_ishida.som.history['z'], htsom_someone.som.history['z'])

        #np.testing.assert_allclose(som_ishida.history['y'],som_watanabe.history['y'])



if __name__ == "__main__":
    unittest.main()
