import unittest

import numpy as np

from libs.models import KernelSmoothing
from libs.models.som_use_for import SOMUseFor


class TestKS(unittest.TestCase):
    def test_invalid_args(self):
        # test about sigma in __init__
        with self.assertRaises(ValueError):
            ks = KernelSmoothing()  # no input
        dummy_args_list = [0, "test_text", np.linspace(-1.0, 1.0, 5), 0.0]
        for arg in dummy_args_list:
            with self.assertRaises(ValueError):
                ks = KernelSmoothing(arg)

        # set correct value
        ks = KernelSmoothing(0.5)

        # prepare
        nb_samples = 50
        input_dim = 3
        output_dim = 2
        seed = 100
        np.random.seed(seed)
        correct_X_1d = np.random.normal(0, 1, nb_samples)
        correct_Y_1d = np.random.normal(0, 1, nb_samples)
        correct_X_2d = np.random.normal(0, 1, (nb_samples, input_dim))
        correct_Y_2d = np.random.normal(0, 1, (nb_samples, output_dim))

        # test about X,Y in fit
        dummy_args_list = [0.0, np.random.normal(0, 1, (nb_samples, 2, 5))]
        for arg in dummy_args_list:
            with self.assertRaises(ValueError):
                ks.fit(X=arg, Y=correct_Y_1d)
            with self.assertRaises(ValueError):
                ks.fit(X=correct_X_1d, Y=arg)

        ## test the case that X and Y are not matched size
        with self.assertRaises(ValueError):
            ks.fit(correct_X_1d, correct_Y_1d.reshape(int(nb_samples / 2), -1))

        # test Xnew in predict
        ## set correct X and Y
        nb_new_samples = 100
        ks.fit(correct_X_1d, correct_Y_1d)
        dummy_Xnew_2d = np.random.normal(0, 1, (nb_new_samples, 2))
        for arg in dummy_args_list:
            with self.assertRaises(ValueError):
                ks.predict(arg)
        ## test the case that X and Xnew are not matched size
        with self.assertRaises(ValueError):
            f = ks.predict(dummy_Xnew_2d)

        # test Xnew in calc_gradient_sqnorm
        for arg in dummy_args_list:
            with self.assertRaises(ValueError):
                grad = ks.calc_gradient_sqnorm(arg)
        ## test the case that X and Xnew are not matched size
        with self.assertRaises(ValueError):
            grad = ks.calc_gradient_sqnorm(dummy_Xnew_2d)

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
