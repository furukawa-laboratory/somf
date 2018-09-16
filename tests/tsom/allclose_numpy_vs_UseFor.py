import unittest

import numpy as np

from libs.models.tsom.tsom import TSOM2
from libs.models.som.som_use_for import SOMUseFor


class TestTSOM2(unittest.TestCase):
    def test_numpy_vs_usefor(self):
        # random seed setting
        seed = 100
        np.random.seed(seed)

        # prepare observed data
        nb_samples1 = 25
        nb_samples2 = 35
        observed_dim = 5

        X = np.random.normal(0, 1, (nb_samples1,nb_samples2,observed_dim))


        nb_epoch = 60
        latent_dim = [1, 2]
        resolution = [20, 30]
        sigma_max = [2.0, 2.2]
        sigma_min = [0.4, 0.2]
        tau = [50, 60]

        # prepare init
        Z1init = np.random.rand(nb_samples1,)
        Z2init =
        som_numpy = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
        som_use_for = SOMUseFor(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)

        som_numpy.fit(nb_epoch=nb_epoch)
        som_use_for.fit(nb_epoch=nb_epoch)

        np.testing.assert_allclose(som_numpy.history['y'],som_use_for.history['y'])

if __name__ == "__main__":
    unittest.main()
