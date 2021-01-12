#TSOM3のペアプロ用のファイル
import unittest
import numpy as np
from libs.models.tsom3 import TSOM3
from tests.tsom.tsom3.tsom3_another import TSOM3_another


class TestTSOM3(unittest.TestCase):
    def test_ishida_vs_another(self):

        # random seed setting
        seed = 100
        np.random.seed(seed)

        # prepare observed data
        nb_samples1 = 5
        nb_samples2 = 15
        nb_samples3 = 20
        nb_observed_dim=4

        X = np.random.normal(0, 1, (nb_samples1, nb_samples2, nb_samples3,nb_observed_dim))

        # set learning parameter
        nb_epoch = 60
        latent_dim = [2, 2,2]
        resolution = [15,10,5]
        sigma_max = [1.0, 2.0,1.5]
        sigma_min = [0.1, 0.2,0.3]
        tau = [15, 25,35]

        ## prepare init
        Z1init = np.random.rand(X.shape[0], latent_dim[0])
        Z2init = np.random.rand(X.shape[1], latent_dim[1])
        Z3init = np.random.rand(X.shape[2], latent_dim[2])
        init = [Z1init, Z2init,Z3init]

        # generate tsom instance
        tsom3_ishida = TSOM3(X, latent_dim=latent_dim, resolution=resolution,
                           SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                           init=init)

        tsom3_another = TSOM3_another(X, latent_dim=latent_dim, resolution=resolution,
                                      SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                                      init=init)
        # learn
        tsom3_ishida.fit(nb_epoch=nb_epoch)
        tsom3_another.fit(nb_epoch=nb_epoch)


        # test
        np.testing.assert_allclose(tsom3_ishida.history['y'], tsom3_another.history['y'])
        np.testing.assert_allclose(tsom3_ishida.history['z1'], tsom3_another.history['z1'])
        np.testing.assert_allclose(tsom3_ishida.history['z2'], tsom3_another.history['z2'])
        np.testing.assert_allclose(tsom3_ishida.history['z3'], tsom3_another.history['z3'])
        np.testing.assert_allclose(tsom3_ishida.history['sigma1'], tsom3_another.history['sigma1'])
        np.testing.assert_allclose(tsom3_ishida.history['sigma2'], tsom3_another.history['sigma2'])
        np.testing.assert_allclose(tsom3_ishida.history['sigma3'], tsom3_another.history['sigma3'])


if __name__ == "__main__":
    unittest.main()
