import unittest

import numpy as np

from libs.models.tsom_plus_som import TSOMPlusSOM
from tests.plus_TSOM.plus_TSOM_watanabe import TSOMPlusSOMWatanabe


class TestTSOMPlusSOM(unittest.TestCase):
    def create_artficial_data(self,n_samples,n_features,n_groups,n_samples_per_group):
        x = np.random.normal(0.0,1.0,(n_samples,n_features))
        if isinstance(n_samples_per_group,int):
            n_samples_per_group = np.ones(n_groups,int) * n_samples_per_group
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
        n_samples_per_group = np.random.randint(1,30,n_groups)  # 各グループにメンバーに何人いるのか
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
                                   group_features=index_members_of_group,
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

    def _transform_list_to_bag(self,list_of_indexes,num_members):
        bag_of_members = np.empty((0,num_members))
        for indexes in list_of_indexes:
            one_hot_vectors = np.eye(num_members)[indexes]
            one_bag = one_hot_vectors.sum(axis=0)[None,:]
            bag_of_members=np.append(bag_of_members,one_bag,axis=0)
        return bag_of_members
    def test_matching_index_member_as_list_or_bag(self):
        seed = 100
        np.random.seed(seed)
        n_members = 100
        n_groups = 10  # group数
        n_features = 3  # 各メンバーの特徴数
        n_samples_per_group = np.random.randint(1,50,n_groups)  # 各グループにメンバーに何人いるのか
        member_features,index_members_of_group = self.create_artficial_data(n_members,
                                                                            n_features,
                                                                            n_groups,
                                                                            n_samples_per_group)
        bag_of_members = self._transform_list_to_bag(index_members_of_group, n_members)

        Z1 = np.random.rand(n_members, 2) * 2.0 - 1.0
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

        tsom_plus_som_input_list = TSOMPlusSOM(member_features=member_features,
                                               group_features=index_members_of_group,
                                               params_tsom=params_tsom,
                                               params_som=params_som)
        tsom_plus_som_input_bag = TSOMPlusSOM(member_features=member_features,
                                              group_features=bag_of_members,
                                              params_tsom=params_tsom,
                                              params_som=params_som)

        tsom_plus_som_input_list.fit(tsom_epoch_num=tsom_epoch_num,
                         kernel_width=kernel_width,
                         som_epoch_num=som_epoch_num)
        tsom_plus_som_input_bag.fit(tsom_epoch_num=tsom_epoch_num,
                           kernel_width=kernel_width,
                           som_epoch_num=som_epoch_num)

        np.testing.assert_allclose(tsom_plus_som_input_list.tsom.history['y'], tsom_plus_som_input_bag.tsom.history['y'])
        np.testing.assert_allclose(tsom_plus_som_input_list.tsom.history['z1'], tsom_plus_som_input_bag.tsom.history['z1'])
        np.testing.assert_allclose(tsom_plus_som_input_list.tsom.history['z2'], tsom_plus_som_input_bag.tsom.history['z2'])
        np.testing.assert_allclose(tsom_plus_som_input_list.params_som['X'], tsom_plus_som_input_bag.params_som['X'])
        np.testing.assert_allclose(tsom_plus_som_input_list.som.history['y'], tsom_plus_som_input_bag.som.history['y'])
        np.testing.assert_allclose(tsom_plus_som_input_list.som.history['z'], tsom_plus_som_input_bag.som.history['z'])

    def test_transform(self):
        # prepare dataset
        seed = 100
        np.random.seed(seed)
        n_members = 1000
        n_groups = 10  # group数
        n_features = 3  # 各メンバーの特徴数
        n_members_per_group = np.random.randint(1,30,n_groups)  # 各グループにメンバーに何人いるのか
        member_features,index_members_of_group = self.create_artficial_data(n_members,
                                                                            n_features,
                                                                            n_groups,
                                                                            n_members_per_group)
        bag_of_members = self._transform_list_to_bag(index_members_of_group,num_members=n_members)

        # prepare parameters
        Z1 = np.random.rand(n_members, 2) * 2.0 - 1.0
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
        kernel_width = 0.3
        som_epoch_num = 50

        # fit
        htsom_bag = TSOMPlusSOM(member_features=member_features,
                            group_features=bag_of_members,
                            params_tsom=params_tsom,
                            params_som=params_som)
        htsom_bag.fit(tsom_epoch_num,kernel_width,som_epoch_num)
        Z_fit_bag = htsom_bag.som.Z
        Z_transformed_bag = htsom_bag.transform(group_features=bag_of_members,kernel_width=kernel_width)

        htsom_list = TSOMPlusSOM(member_features=member_features,
                                group_features=index_members_of_group,
                                params_tsom=params_tsom,
                                params_som=params_som)
        htsom_list.fit(tsom_epoch_num,kernel_width,som_epoch_num)
        Z_fit_list = htsom_list.som.Z
        Z_transformed_list = htsom_list.transform(group_features=index_members_of_group,kernel_width=kernel_width)

        # compare estimated latent variables in fit and one in transform
        np.testing.assert_allclose(Z_fit_bag,Z_transformed_bag)
        np.testing.assert_allclose(Z_fit_list,Z_transformed_list)


if __name__ == "__main__":
    unittest.main()
