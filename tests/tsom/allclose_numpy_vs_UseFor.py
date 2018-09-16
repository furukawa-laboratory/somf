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

    def test_arg_in_constructor(self):
        # random seed setting
        seed = 100
        np.random.seed(seed)

        # prepare observed data
        nb_samples1 = 25
        nb_samples2 = 35
        observed_dim = 1

        X_3d = np.random.normal(0, 1, (nb_samples1, nb_samples2, observed_dim))
        X_2d = X.reshape(nb_samples1,nb_samples2)

        # set learning parameter
        nb_epoch = 60

        latent_dim = [1, 1]
        resolution = [20, 20]
        sigma_max = [2.0, 2.0]
        sigma_min = [0.4, 0.4]
        tau = [50, 50]

        ## prepare init
        Z1init = np.random.rand(nb_samples1, latent_dim[0])
        Z2init = np.random.rand(nb_samples2, latent_dim[1])
        init = [Z1init, Z2init]

        tsom_type1 = TSOM2(X=X_3d, latent_dim=latent_dim, resolution=resolution,
                           SIGMA_MAX=sigma_max,SIGMA_MIN=sigma_min,TAU=tau,init=init)

        tsom_type2 = TSOM2(X=X_2d, latent_dim=latent_dim[0], resolution=resolution[0],
                           SIGMA_MAX=sigma_max[0],SIGMA_MIN=sigma_min[0],TAU=tau[0],init=init)

        tsom_type1.fit(nb_epoch=nb_epoch)
        tsom_type2.fit(nb_epoch=nb_epoch)



if __name__ == "__main__":
    unittest.main()
