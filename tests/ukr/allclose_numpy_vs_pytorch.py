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
        # torch.manual_seed(seed)
        X = create_data(nb_samples)
        x_sigma = 0.1
        X += np.random.normal(0, x_sigma, X.shape)

        # common parameter
        n_components = 2
        bandwidth_gaussian_kernel = 0.2
        nb_epoch = 1

        # ukr parameter
        is_compact = True
        is_save_history = True
        lambda_ = 0.0
        eta = 8.0

        # Zinit
        Zinit = np.random.rand(nb_samples, n_components) * 2.0 - 1.0

        ukr = UKR(X, n_components=n_components, bandwidth_gaussian_kernel=bandwidth_gaussian_kernel,
                is_compact=is_compact, is_save_history=is_save_history, lambda_=lambda_, init=Zinit)
        ukr.fit(nb_epoch=nb_epoch, eta=eta)
        z = ukr.history['z'][0]
        y = ukr.history['y'][0]
        ukr.calculation_history_of_mapping(resolution=30)

        ukr_pytorch = UKR_pytorch(torch.from_numpy(X), nb_components=n_components, bandwidth_gaussian_kernel=bandwidth_gaussian_kernel,
                is_compact=is_compact, is_save_history=is_save_history, lambda_=lambda_,
                init=torch.tensor(Zinit, requires_grad=True, dtype=torch.float64))
        ukr_pytorch.fit(nb_epoch=nb_epoch, eta=eta)
        z_pytorch = ukr_pytorch.history['z'][0].detach().numpy()
        y_pytorch = ukr_pytorch.history['y'][0].detach().numpy()
        # np.testing.assert_allclose(y, y_pytorch, atol=1e-06)
        np.testing.assert_allclose(z, z_pytorch, atol=1e-06)

if __name__ == "__main__":
    # unittest.main()
    test_numpy_vs_pytorch()
