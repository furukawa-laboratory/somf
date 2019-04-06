import unittest

import numpy as np

from libs.models.som import SOM
from  tests.som.allclose_somkl.som_someone import SOM_someone


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
        Zinit = np.random.rand(N,L)*2.0 -1.0

        nb_epoch = 200
        SIGMA_MAX = 2.2
        SIGMA_MIN = 0.1
        TAU = 50
        som_ishida = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
        som_someone = SOM_someone(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)

        som_ishida.fit(nb_epoch=nb_epoch,euclid=True)
        som_someone.fit(nb_epoch=nb_epoch,euclid=True)


        np.testing.assert_allclose(som_ishida.history['y'],som_someone.history['y'])



if __name__ == "__main__":
    unittest.main()
