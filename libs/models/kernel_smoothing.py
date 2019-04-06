# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance as dist


class KernelSmoothing:
    def __init__(self, sigma=None):
        if sigma is not None:
            pass
        else:
            raise ValueError("sigma must be input:{}".format(sigma))

        if isinstance(sigma, float) and sigma > 0.0:
            self.sigma = sigma
        else:
            raise ValueError("invalid sigma:{}".format(sigma))

    def _check_correct_ndarray(self, array, array_name):
        if isinstance(array, np.ndarray):
            if array.ndim == 1 or array.ndim == 2:
                pass
            else:
                raise ValueError("invalid {}: {}".format(array_name, array))
        else:
            raise ValueError("invalid {}: {}".format(array_name, array))

    def fit(self, X, Y):
        # check formats of X and Y
        for array, array_name in zip([X, Y], ["X", "Y"]):
            self._check_correct_ndarray(array, array_name)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be same size of 1st mode")

        # set X and Y as 2d array
        nb_samples = X.shape[0]
        self.X = X.copy().reshape(nb_samples, -1)
        self.Y = Y.copy().reshape(nb_samples, -1)
        self.input_dim = self.X.shape[1]

    def predict(self, Xnew):
        # check format of Xnew
        self._check_correct_ndarray(Xnew, "Xnew")
        Xnew = Xnew.reshape(Xnew.shape[0], -1)
        if Xnew.shape[1] != self.input_dim:
            raise ValueError("X and Xnew must be same dimension")

        # calculate output of Xnew
        R = self._calc_standardized_coefficient(Xnew)
        F = R @ self.Y  # calculate output value

        return F

    def _calc_standardized_coefficient(self, Xnew):
        Dist = dist.cdist(Xnew, self.X, 'sqeuclidean')
        H = np.exp(-Dist / (2 * self.sigma * self.sigma))  # calculate values of kernel function
        G = np.sum(H, axis=1)[:, np.newaxis]  # sum along with 1th mode
        Ginv = np.reciprocal(G)  # calculate reciprocal of G
        R = H * Ginv  # standardize to sum is 1.0

        return R

    def calc_gradient(self, Xnew):
        # check format of Xnew
        self._check_correct_ndarray(Xnew, "Xnew")
        Xnew = Xnew.reshape(Xnew.shape[0], -1)
        if Xnew.shape[1] != self.input_dim:
            raise ValueError("X and Xnew must be same dimension")

        # calculate gradient of Xnew (size: Xnew.shape[0] x output_dim x input_dim)
        R = self._calc_standardized_coefficient(Xnew)
        V = R[:, :, np.newaxis] * (self.X[np.newaxis, :, :] - Xnew[:, np.newaxis, :]) / (
                    self.sigma * self.sigma)  # KxNxL
        V_mean = V.sum(axis=1)[:, np.newaxis, :]  # Kx1xL

        # calculate true gradient squared norm
        dRdX = V - R[:, :, np.newaxis] * V_mean  # KxNxL
        dFdX = np.einsum("knl,nd->kdl", dRdX, self.Y)  # KxDxL
        return dFdX

    def calc_gradient_sqnorm(self, Xnew):
        # calculate gradient squared norm of Xnew
        dFdX = self.calc_gradient(Xnew)
        return np.sum(dFdX ** 2, axis=(1, 2))  # K
