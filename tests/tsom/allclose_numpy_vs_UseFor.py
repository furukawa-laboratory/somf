import unittest

import numpy as np

from libs.models.tsom.tsom import TSOM2
from libs.models.tsom.tsom_use_for import TSOM2UseFor


class TestTSOM2(unittest.TestCase):
    def test_numpy_vs_usefor(self):
        # random seed setting
        seed = 100
        np.random.seed(seed)

        # prepare observed data
        nb_samples1 = 25
        nb_samples2 = 35
        observed_dim = 5

        X = np.random.normal(0, 1, (nb_samples1, nb_samples2, observed_dim))

        # set learning parameter
        nb_epoch = 60
        latent_dim = [1, 2]
        resolution = [20, 30]
        sigma_max = [2.0, 2.2]
        sigma_min = [0.4, 0.2]
        tau = [50, 60]

        ## prepare init
        Z1init = np.random.rand(nb_samples1, latent_dim[0])
        Z2init = np.random.rand(nb_samples2, latent_dim[1])
        init = [Z1init, Z2init]

        # generate tsom instance
        tsom_numpy = TSOM2(X, latent_dim=latent_dim, resolution=resolution,
                           SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                           init=init)
        tsom_use_for = TSOM2UseFor(X, latent_dim=latent_dim, resolution=resolution,
                                   SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                                   init=init)

        # learn
        tsom_numpy.fit(nb_epoch=nb_epoch)
        tsom_use_for.fit(nb_epoch=nb_epoch)

        # test
        np.testing.assert_allclose(tsom_numpy.history['y'], tsom_use_for.history['y'])
        np.testing.assert_allclose(tsom_numpy.history['z1'], tsom_use_for.history['z1'])
        np.testing.assert_allclose(tsom_numpy.history['z2'], tsom_use_for.history['z2'])


if __name__ == "__main__":
    unittest.main()
