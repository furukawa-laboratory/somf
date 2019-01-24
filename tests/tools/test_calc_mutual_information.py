import unittest
import numpy as np

from libs.tools import calc_mutual_information


class TestCalcMutualInformation(unittest.TestCase):
    def test_one_and_zero(self):
        nb_samples = 1000000
        seed = 100
        nb_bins = 100
        bias = 0.0

        np.random.rand(seed)

        x = np.linspace(-1.0, 1.0, nb_samples)
        y1 = x
        y2 = np.random.rand(nb_samples)

        MI1 = calc_mutual_information(x,y1,
                                      nb_bins=nb_bins,
                                      bias=bias,
                                      normalize=True)
        MI2 = calc_mutual_information(x,y2,
                                      nb_bins=nb_bins,
                                      bias=bias,
                                      normalize=True)

        np.testing.assert_allclose(MI1,1.0,atol=1e-7)
        np.testing.assert_allclose(MI2,0.0,atol=1e-2)

    def test_args(self):

        x = np.random.rand(100,2)
        y = np.random.rand(100,4)
        with self.assertRaises(ValueError):
            calc_mutual_information(x=x,y=y)

        x = np.random.rand(100)
        y = np.random.rand(101)
        with self.assertRaises(ValueError):
            calc_mutual_information(x=x,y=y)




if __name__ == "__main__":
    unittest.main()
