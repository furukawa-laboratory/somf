import unittest

import numpy as np

from libs.models import KernelSmoothing
from libs.models.som_use_for import SOMUseFor


class TestKS(unittest.TestCase):
    def test_invalid_args(self):
        # test about sigma
        with self.assertRaises(ValueError):
            ks=KernelSmoothing()
            args_list = [0, "test_text", np.linspace(-1.0,1.0,5), 0.0]
            for arg in args_list:
                ks=KernelSmoothing(arg)
    # def test_numpy_vs_usefor(self):
    #     N = 100
    #     D = 3
    #     L = 2
    #     resolution = 10
    #     M = resolution ** L
    #     seed = 100
    #     np.random.seed(seed)
    #     X = np.random.normal(0, 1, (N, D))
    #     Zinit = np.random.rand(N,L)
    #
    #     nb_epoch = 200
    #     SIGMA_MAX = 2.2
    #     SIGMA_MIN = 0.1
    #     TAU = 50
    #     som_numpy = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
    #     som_use_for = SOMUseFor(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
    #
    #     som_numpy.fit(nb_epoch=nb_epoch)
    #     som_use_for.fit(nb_epoch=nb_epoch)
    #
    #     np.testing.assert_allclose(som_numpy.history['y'],som_use_for.history['y'])

if __name__ == "__main__":
    unittest.main()
