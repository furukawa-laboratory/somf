import unittest

import numpy as np
from scipy import sparse

from libs.models.tsom import TSOM2 as TSOM2Dense
from libs.models.tsom_sparse import TSOM2 as TSOM2Sparse


class TestTSOM2(unittest.TestCase):
    def test_dense_vs_sparse(self):
        # random seed setting
        seed = 100
        np.random.seed(seed)

        # prepare observed data
        nb_samples1 = 10
        nb_samples2 = 20
        # observed_dim = 1

        X = np.random.randint(0, 3, (nb_samples1, nb_samples2))
        X_sparse = sparse.csr_matrix(sparse.lil_matrix(X))

        # set learning parameter
        nb_epoch = 60
        latent_dim = [1, 2]
        resolution = [7, 9]
        sigma_max = [2.0, 2.2]
        sigma_min = [0.4, 0.2]
        tau = [50, 60]

        # prepare init
        Z1init = np.random.rand(nb_samples1, latent_dim[0])
        Z2init = np.random.rand(nb_samples2, latent_dim[1])
        init = [Z1init, Z2init]

        # generate tsom instance
        tsom_dense = TSOM2Dense(X, latent_dim=latent_dim, resolution=resolution,
                                SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                                init=init)
        tsom_sparse = TSOM2Sparse(X_sparse, latent_dim=latent_dim, resolution=resolution,
                                  sigma_max=sigma_max, sigma_min=sigma_min, tau=tau,
                                  init=init, sigma_mode="LINEAR")

        # learn
        tsom_dense.fit(nb_epoch=nb_epoch)
        tsom_sparse.fit(nb_epoch=nb_epoch, is_direct=False)

        # test
        np.testing.assert_allclose(tsom_dense.history['y'][:, :, :, 0], tsom_sparse.history['y'])
        np.testing.assert_allclose(tsom_dense.history['z1'], tsom_sparse.history['z1'])
        np.testing.assert_allclose(tsom_dense.history['z2'], tsom_sparse.history['z2'])


if __name__ == "__main__":
    unittest.main()
