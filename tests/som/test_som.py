import unittest

import numpy as np

from libs.models.som import SOM
from libs.models.som_use_for import SOMUseFor


class TestSOM(unittest.TestCase):
    def test_numpy_vs_usefor(self):
        N = 100
        D = 3
        L = 2
        resolution = 10
        M = resolution ** L
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Zinit = np.random.rand(N,L)

        nb_epoch = 200
        SIGMA_MAX = 2.2
        SIGMA_MIN = 0.1
        TAU = 50
        som_numpy = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
        som_use_for = SOMUseFor(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)

        som_numpy.fit(nb_epoch=nb_epoch)
        som_use_for.fit(nb_epoch=nb_epoch)

        np.testing.assert_allclose(som_numpy.history['y'],som_use_for.history['y'])

    def test_init(self):
        N = 100
        D = 3
        L = 2
        resolution = 10
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Zinit = np.random.rand(N,L)

        inits = ['random', 'random_bmu', Zinit]

        SIGMA_MAX = 2.2
        SIGMA_MIN = 0.1
        TAU = 50

        for init in inits:
            som = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=init)



if __name__ == "__main__":
    unittest.main()
