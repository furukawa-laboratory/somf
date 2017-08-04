import unittest

import numpy as np

from som import SOM
from som_use_for import SOMUseFor

from tqdm import tqdm

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

        T = 200
        SIGMA_MAX = 2.2
        SIGMA_MIN = 0.1
        TAU = 50
        som_numpy = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
        som_use_for = SOMUseFor(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)


        allY_numpy = np.zeros((M, D, T))
        allY_use_for = np.zeros((M, D, T))

        for t in tqdm(range(T)):
            som_numpy.learning(t)
            allY_numpy[:, :, t] = som_numpy.Y
        for t in tqdm(range(T)):
            som_use_for.learning(t)
            allY_use_for[:, :, t] = som_use_for.Y

        np.testing.assert_allclose(allY_numpy,allY_use_for)

if __name__ == "__main__":
    unittest.main()
