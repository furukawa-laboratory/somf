import unittest

import numpy as np
import itertools

from lib.models.iwasaki.KSE0331 import KSE as KSE_Iwasaki
from lib.models.watanabe.KSE0331_add_changing_beta import KSE as KSE_Watanabe


class TestKSE0331(unittest.TestCase):
    def test_iwasaki_watanabe(self):
        N = 100
        D = 3
        L = 2
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Zrand = np.random.normal(0, 0.1, (N, L))
        Zs = ['random', Zrand]
        betaTypes = ['type1', 'type2']

        nb_epoch = 100
        epsilon = 0.5
        gamma = 1.0
        sigma = 30.0

        for Z0, betaType in itertools.product(Zs, betaTypes):
            with self.subTest(betaType=betaType, Z0=Z0):
                np.random.seed(seed)
                kse_iwa = KSE_Iwasaki(X, L, Z0, betaType=betaType)
                np.random.seed(seed)
                kse_wata = KSE_Watanabe(X, L, Z0, choice_beta=betaType)

                kse_iwa.fit(nb_epoch=nb_epoch, epsilon=epsilon, gamma=gamma, sigma=sigma)
                kse_wata.fit(nb_epoch=nb_epoch, epsilon=epsilon, sigma=sigma)

                np.testing.assert_allclose(kse_iwa.history['z'], kse_wata.history['z'])

if __name__ == "__main__":
    unittest.main()
