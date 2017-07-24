import unittest

import numpy as np

from lib.models.flab.KSE0524 import KSE as KSE_Flab
from lib.models.iwasaki.KSE0524 import KSE as KSE_Iwasaki


class TestKSE0428(unittest.TestCase):
    def test_flab_iwasaki(self):
        N = 100
        D = 3
        L = 2
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Z0 = np.random.normal(0, 0.1, (N, L))
        kse_flab = KSE_Flab(X, L, Z0)
        kse_iwa = KSE_Iwasaki(X, L, Z0)

        Epoch = 1000
        epsilon = 0.5
        kse_flab.fit(nb_epoch=Epoch, epsilon=epsilon)
        kse_iwa.fit(nb_epoch=Epoch, z_epsilon=epsilon)

        np.testing.assert_allclose(kse_iwa.history['z'], kse_flab.history['z'], atol=1e-8, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
