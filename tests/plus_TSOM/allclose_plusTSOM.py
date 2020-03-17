import unittest

import numpy as np

from libs.models.tsom_plus_som import TSOMPlusSOM
from tests.plus_TSOM.plus_TSOM_watanabe import TSOMPlusSOMWatanabe


class TestTSOMPlusSOM(unittest.TestCase):
    def create_artficial_data(self,n_samples,n_features,n_groups,n_samples_per_group):
        x = np.random.normal(0.0,1.0,(n_samples,n_features))
        if isinstance(n_samples_per_group,int):
            index_members_of_group = np.random.randint(0,n_samples,(n_groups,n_samples_per_group))
        elif isinstance(n_samples_per_group,np.ndarray):
            index_members_of_group = []
            for n_samples_in_the_group in n_samples_per_group:
                index_members_of_group.append(np.random.randint(0,n_samples,n_samples_in_the_group))
        return x, index_members_of_group

    def test_plusTSOM_ishida_vs_test_plusTSOM_watanabe(self):
        seed = 100
        np.random.seed(seed)
        n_samples = 1000
        n_groups = 10  # group数
        n_features = 3  # 各メンバーの特徴数
        n_samples_per_group = 30  # 各グループにメンバーに何人いるのか
        member_features,index_members_of_group = self.create_artficial_data(n_samples,
                                                                           n_features,
                                                                           n_groups,
                                                                           n_samples_per_group)
        # 1stTSOMの初期値
        Z1 = np.random.rand(n_samples, 2) * 2.0 - 1.0
        Z2 = np.random.rand(n_features, 2) * 2.0 - 1.0
        init_TSOM = [Z1, Z2]
        init_SOM = np.random.rand(n_groups, 2) * 2.0 - 1.0


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
