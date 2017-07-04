import unittest

import numpy as np

from lib.models.flab.KSE0331 import KSE as KSE_Flab
from lib.models.iwasaki.KSE0331 import KSE as KSE_Iwasaki
from lib.models.watanabe.KSE170331_wata_gamma import KSE170331_wata_gamma as KSE_Watanabe


class TestKSE0331(unittest.TestCase):
    def test_iwasaki_watanabe(self):
        N = 100
        D = 3
        L = 2
        M = 100
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Z0 = np.random.normal(0, 0.1, (N, L))
        kse_iwa = KSE_Iwasaki(X, L, Z0)
        kse_wata = KSE_Watanabe(X, L, M, Z0)

        Epoch = 100
        epsilon = 0.5
        gamma = 1.0
        sigma = 30.0
        alpha = 1 / (sigma ** 2)
        kse_iwa.fit(nb_epoch=Epoch, epsilon=epsilon, gamma=gamma, sigma=sigma)
        kse_wata.fit(Epoch=Epoch, epsilon=epsilon, alpha=alpha)

        np.testing.assert_allclose(kse_iwa.history['z'], kse_wata.history['z'])

    def test_flab_watanabe(self):
        N = 100
        D = 3
        L = 2
        M = 100
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Z0 = np.random.normal(0, 0.1, (N, L))
        kse_flab = KSE_Flab(X, L, Z0)
        kse_wata = KSE_Watanabe(X, L, M, Z0)

        Epoch = 100
        epsilon = 0.5
        gamma = 1.0
        sigma = 30.0
        alpha = 1 / (sigma ** 2)
        kse_flab.fit(nb_epoch=Epoch, epsilon=epsilon, gamma=gamma, sigma=sigma)
        kse_wata.fit(Epoch=Epoch, epsilon=epsilon, alpha=alpha)

        np.testing.assert_allclose(kse_flab.history['z'], kse_wata.history['z'])

if __name__ == "__main__":
    unittest.main()
