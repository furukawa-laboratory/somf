import unittest
import numpy as np
from scipy.stats import invwishart

from libs.tools import calc_mutual_information


class TestCalcMutualInformation(unittest.TestCase):
    def test_one_and_zero(self):
        nb_samples = 1000000
        seed = 100
        nb_bins = 100
        bias = 0.0

        np.random.seed(seed)

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

    def test_compare_analytical_value(self):
        nb_samples = 100000
        seed = 500
        nb_patterns = 10

        nb_bins = 1000
        bias = 0.0001

        np.random.seed(seed)

        mean = np.zeros(2)
        df = 3.0
        scale = 2.0 * np.array([[1.0,0.0],[0.0,1.0]])
        iw = invwishart(df=df,scale=scale)
        covs = iw.rvs(nb_patterns)

        for cov in covs:
            x_2d = np.random.multivariate_normal(mean,cov,nb_samples)

            NMI_func = calc_mutual_information(x_2d[:,0],x_2d[:,1],
                                               nb_bins=nb_bins,
                                               bias=bias,
                                               normalize=True)
            rho = cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])
            MI_analytical = -0.5 * np.log(1.0 - (rho**2.0))
            entropy0 = 0.5 * (1.0 + np.log(cov[0,0]) + np.log(2.0*np.pi))
            entropy1 = 0.5 * (1.0 + np.log(cov[1,1]) + np.log(2.0*np.pi))
            NMI_analytical = MI_analytical / (0.5 * (entropy0+entropy1))
            print('func={},analy={}'.format(NMI_func,NMI_analytical))




if __name__ == "__main__":
    unittest.main()
