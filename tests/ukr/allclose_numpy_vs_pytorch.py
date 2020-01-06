import unittest

import numpy as np
import torch

from libs.datasets.artificial.kura import create_data

from libs.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as UKR
from libs.models.unsupervised_kernel_regression_pytorch import Unsupervised_Kernel_Regression_pytorch as UKR_pytorch


class TestUKR(unittest.TestCase):
    def test_numpy_vs_pytorch(self):
        # create artiricial dataset
        nb_samples = 500
        seed = 1
        np.random.seed(seed)
        X = create_data(nb_samples)
        x_sigma = 0.1
        X += np.random.normal(0, x_sigma, X.shape)

        # set parameter
        n_components = 2
        bandwidth_gaussian_kernel_list = [0.2, 1.0]
        nb_epoch = 500

        is_compact_list = [True, False]
        is_save_history = True
        lambda_ = 0.0
        eta = 8.0

        # initialize Z
        Zinit = np.random.rand(nb_samples, n_components) * 2.0 - 1.0

        for bandwidth_gaussian_kernel in bandwidth_gaussian_kernel_list:
            for is_compact in is_compact_list:
                ukr = UKR(X, n_components=n_components, bandwidth_gaussian_kernel=bandwidth_gaussian_kernel,
                          is_compact=is_compact, is_save_history=is_save_history, lambda_=lambda_, init=Zinit)
                ukr.fit(nb_epoch=nb_epoch, eta=eta)
                all_z = ukr.history['z']

                ukr_pytorch = UKR_pytorch(torch.from_numpy(X),
                                          nb_components=n_components,
                                          bandwidth_gaussian_kernel=bandwidth_gaussian_kernel,
                                          is_compact=is_compact, is_save_history=is_save_history, lambda_=lambda_,
                                          init=torch.tensor(Zinit, requires_grad=True, dtype=torch.float64))
                ukr_pytorch.fit(nb_epoch=nb_epoch, eta=eta)
                all_z_pytorch = ukr_pytorch.history['z'].detach().numpy()
                np.testing.assert_allclose(all_z, all_z_pytorch)


if __name__ == "__main__":
    unittest.main()
