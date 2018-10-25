import unittest

import numpy as np

from libs.models.som import SOM
from libs.models.som_tensorflow import SOM as som


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
        som_numpy = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
        som_tensorflow = som(X.shape[1], X.shape[0], n=resolution, m=resolution,epochs=nb_epoch,sigma_max=SIGMA_MAX,sigma_min=SIGMA_MIN,tau=TAU,init=Zinit)

        som_numpy.fit(nb_epoch=nb_epoch)
        som_tensorflow.predict(X)



        np.testing.assert_allclose(som_numpy.history['y'],som_tensorflow.historyY, atol=1e-06)



if __name__ == "__main__":
    unittest.main()
