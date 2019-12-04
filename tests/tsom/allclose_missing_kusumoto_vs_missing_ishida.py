import unittest

import numpy as np

from libs.models.tsom import TSOM2
from libs.models.tsom2_ishida import TSOM2_ishida


class TestTSOM_missing(unittest.TestCase):
    def test_kusumoto_vs_ishida(self):
        # random seed setting
        seed = 100
        np.random.seed(seed)

        # prepare observed data
        nb_samples1 = 10
        nb_samples2 = 20
        observed_dim = 3

        X = np.random.normal(0, 1, (nb_samples1, nb_samples2, observed_dim))
        #gammaの生成
        gamma=np.random.rand(nb_samples1, nb_samples2, observed_dim)
        for i in np.arange(nb_samples1):
            for j in np.arange(nb_samples2):
                for k in np.arange(observed_dim):
                    if gamma[i,j,k]>=0.5:
                        gamma[i,j,k]=1
                    elif gamma[i,j,k]<0.5:
                        gamma[i, j,k] = 0
        # set learning parameter
        nb_epoch = 250
        latent_dim = [1, 1]
        resolution = [7, 9]
        sigma_max = [2.0, 2.2]
        sigma_min = [0.4, 0.2]
        tau = [50, 60]

        ## prepare init
        Z1init = np.random.rand(nb_samples1, latent_dim[0])
        Z2init = np.random.rand(nb_samples2, latent_dim[1])
        init = [Z1init, Z2init]

        # generate tsom instance
        tsom_kusumoto = TSOM2(X, latent_dim=latent_dim, resolution=resolution,
                           SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                           init=init,model = 'direct',gamma=gamma)
        tsom_ishida = TSOM2_ishida(X, latent_dim=latent_dim, resolution=resolution,
                                   SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                                   init=init,model = 'direct',gamma=gamma)

        # learn
        tsom_kusumoto.fit(nb_epoch=nb_epoch)
        tsom_ishida.fit(nb_epoch=nb_epoch)

        # test
        np.testing.assert_allclose(tsom_kusumoto.history['y'], tsom_ishida.history['y'],rtol=1e-09)
        np.testing.assert_allclose(tsom_kusumoto.history['z1'], tsom_ishida.history['z1'])
        np.testing.assert_allclose(tsom_kusumoto.history['z2'], tsom_ishida.history['z2'])

if __name__ == "__main__":
    unittest.main()
