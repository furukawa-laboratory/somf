import unittest

import numpy as np

from libs.models.TSOMPlusSOM import TSOMPlusSOM
from tests.plus_TSOM.plus_TSOM_watanabe import TSOMPlusSOMWatanabe


class TestSOM(unittest.TestCase):
    def test_plusTSOM_ishida_vs_test_plusTSOM_someone(self):
        # 学習データの作成-------------------------------------------
        n_group = 10  # group数
        n_features = 3  # 各メンバーの特徴数
        n_samples_per_group = 30  # 各グループにメンバーに何人いるのか
        seed = 100
        np.random.seed(seed)
        # 1stTSOMの初期値
        Z1 = np.random.rand(n_group * n_samples_per_group, 2) * 2.0 - 1.0
        Z2 = np.random.rand(n_features, 2) * 2.0 - 1.0
        init_TSOM = [Z1, Z2]
        init_SOM = np.random.rand(n_group, 2) * 2.0 - 1.0

        # 学習データの用意
        mean = np.random.rand(n_group, n_features)
        member_features = np.zeros((n_group, n_samples_per_group, n_features))

        for i in range(n_group):
            samples = np.random.multivariate_normal(mean=mean[i], cov=np.identity(n_features), size=n_samples_per_group)
            member_features[i, :, :] = samples

        member_features = member_features.reshape((n_group * n_samples_per_group, n_features))
        index_members_of_group = np.zeros((n_group, n_samples_per_group), dtype=int)

        params_tsom = {'latent_dim': [2, 2],
                       'resolution': [10, 10],
                       'SIGMA_MAX': [1.0, 1.0],
                       'SIGMA_MIN': [0.1, 0.1],
                       'TAU': [50, 50],
                       'init': init_TSOM}
        params_som = {'latent_dim': 2,
                      'resolution': 10,
                      'sigma_max': 2.0,
                      'sigma_min': 0.5,
                      'tau': 50,
                      'init': init_SOM}
        tsom_epoch_num = 50
        som_epoch_num = 50
        kernel_width = 0.3

        htsom_ishida = TSOMPlusSOM(member_features=member_features,
                                   index_members_of_group=index_members_of_group,
                                   params_tsom=params_tsom,
                                   params_som=params_som)
        htsom_watanabe = TSOMPlusSOMWatanabe(member_features=member_features,
                                             index_members_of_group=index_members_of_group,
                                             params_tsom=params_tsom,
                                             params_som=params_som)

        htsom_ishida.fit(tsom_epoch_num=tsom_epoch_num,
                         kernel_width=kernel_width,
                         som_epoch_num=som_epoch_num)
        htsom_watanabe.fit(tsom_epoch_num=tsom_epoch_num,
                           kernel_width=kernel_width,
                           som_epoch_num=som_epoch_num)

        np.testing.assert_allclose(htsom_ishida.tsom.history['y'], htsom_watanabe.tsom.history['y'])
        np.testing.assert_allclose(htsom_ishida.tsom.history['z1'], htsom_watanabe.tsom.history['z1'])
        np.testing.assert_allclose(htsom_ishida.tsom.history['z2'], htsom_watanabe.tsom.history['z2'])
        np.testing.assert_allclose(htsom_ishida.params_som['X'], htsom_watanabe.params_som['X'])
        np.testing.assert_allclose(htsom_ishida.som.history['y'], htsom_watanabe.som.history['y'])
        np.testing.assert_allclose(htsom_ishida.som.history['z'], htsom_watanabe.som.history['z'])


if __name__ == "__main__":
    unittest.main()
