import unittest

import numpy as np

from lib.models.iwasaki.KSE0331 import KSE as KSE_IWA
from lib.models.watanabe.KSE0331 import KSE as KSE_WATA


class TestKSE0331(unittest.TestCase):
    def test_iwasaki_watanabe(self):
        kse_iwa = KSE_IWA()
        kse_wata = KSE_WATA()
        np.testing.assert_allclose(kse_iwa.z, kse_wata.z)


if __name__ == "__main__":
    unittest.main()
