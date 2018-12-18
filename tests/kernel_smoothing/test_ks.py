import unittest

import numpy as np
from scipy.spatial import distance as dist

from libs.models import KernelSmoothing
from libs.models import KernelSmoothingTheano
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

    def test_macthing_results_ks_in_som_and_som_use_for(self):
        N = 100
        D = 3
        L = 2
        resolution = 10
        M = resolution ** L
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Zinit = np.random.rand(N,L)

        nb_epoch = 50
        SIGMA_MAX = 2.2
        SIGMA_MIN = 0.1
        TAU = 50

        # learn som_use_for
        som_use_for = SOMUseFor(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
        som_use_for.fit(nb_epoch=nb_epoch)


        # calculate som algorithm using KernelSmoothing
        Zeta = som_use_for.Zeta.copy()
        Z = Zinit
        historyY = np.zeros((nb_epoch,M,D))
        for epoch in range(nb_epoch):
            sigma = max(SIGMA_MIN, SIGMA_MAX * ( 1 - (epoch / TAU) ) )
            ks = KernelSmoothing(sigma=sigma)
            ks.fit(X=Z, Y=X)
            Y = ks.predict(Zeta)
            Dist = dist.cdist(X, Y)
            bmus = Dist.argmin(axis=1)
            Z = Zeta[bmus, :]
            historyY[epoch] = Y

        # compare two results
        np.testing.assert_allclose(historyY,som_use_for.history['y'])

    def test_matching_theano(self):
        nb_samples = 500
        nb_new_samples = 750
        input_dim = 5
        output_dim = 1 # Theano ver doesn't follow output_dim > 1 in calculation of gradient
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0.0,1.0,(nb_samples,input_dim))
        Y = np.random.normal(0.0,1.0,(nb_samples,output_dim))
        Xnew = np.random.normal(0.0,1.0,(nb_new_samples,input_dim))

        sigma = 0.2
        ks_numpy = KernelSmoothing(sigma)
        ks_numpy.fit(X,Y)
        f_numpy = ks_numpy.predict(Xnew)
        grad_numpy = ks_numpy.calc_gradient(Xnew)
        grad_sq_norm_numpy = ks_numpy.calc_gradient_sqnorm(Xnew)

        ks_theano = KernelSmoothingTheano(sigma)
        ks_theano.fit(X,Y)
        f_theano = ks_theano.predict(Xnew)
        grad_theano = ks_theano.calc_gradient(Xnew)
        grad_sq_norm_theano = ks_theano.calc_gradient_sqnorm(Xnew)

        np.testing.assert_allclose(f_numpy,f_theano)
        np.testing.assert_allclose(grad_numpy.reshape(nb_new_samples,input_dim),grad_theano,atol=1e-14)# default value is atol=0.0
        np.testing.assert_allclose(grad_sq_norm_numpy,grad_sq_norm_theano,atol=1e-20)# default value is atol=0.0
        # Because np.std(grad_theano) = 2.5377, the different seems to be very small value.



if __name__ == "__main__":
    unittest.main()
