# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance as dist
from libs.models import KernelSmoothing
import theano
import theano.tensor as tt


class KernelSmoothingTheano(KernelSmoothing):
    def fit(self, X, Y):
        super(KernelSmoothingTheano, self).fit(X, Y)
        self.X = theano.shared(self.X.copy())
        self.Y = theano.shared(self.Y.copy())

    def _difine_ks(self):
        xnew = tt.dmatrix('xnew')
        xnew1 = xnew.dimshuffle(0, 'x', 1)
        Delta = xnew1 - self.X[np.newaxis,:,:]
        Dist = tt.sum(tt.square(Delta), axis=2)
        H = tt.exp(-0.5*Dist/(self.sigma*self.sigma))
        G = tt.sum(H, axis=1).dimshuffle(0, 'x')
        GInv = 1.0 / G
        R = H * GInv
        F = tt.dot(R, self.Y)
        mapping = theano.function(inputs=[xnew], outputs=[F])

        return mapping
    def predict(self, Xnew):
        # check format of Xnew
        self._check_correct_ndarray(Xnew, "Xnew")
        Xnew = Xnew.reshape(Xnew.shape[0], -1)
        if Xnew.shape[1] != self.input_dim:
            raise ValueError("X and Xnew must be same dimension")

        mapping = self._difine_ks()

        fnew, = mapping(Xnew)
        return fnew

    # def calc_gradient(self, Xnew):


    # def _calc_standardized_coefficient(self, Xnew):
    #     Dist = dist.cdist(Xnew, self.X, 'sqeuclidean')
    #     H = np.exp(-Dist / (2 * self.sigma * self.sigma))  # calculate values of kernel function
    #     G = np.sum(H, axis=1)[:, np.newaxis]  # sum along with 1th mode
    #     Ginv = np.reciprocal(G)  # calculate reciprocal of G
    #     R = H * Ginv  # standardize to sum is 1.0
    #
    #     return R

    # def calc_gradient(self, Xnew):
    #     # check format of Xnew
    #     self._check_correct_ndarray(Xnew, "Xnew")
    #     Xnew = Xnew.reshape(Xnew.shape[0], -1)
    #     if Xnew.shape[1] != self.input_dim:
    #         raise ValueError("X and Xnew must be same dimension")
    #
    #     # calculate gradient of Xnew (size: Xnew.shape[0] x output_dim x input_dim)
    #     R = self._calc_standardized_coefficient(Xnew)
    #     V = R[:, :, np.newaxis] * (self.X[np.newaxis, :, :] - Xnew[:, np.newaxis, :])  # KxNxL
    #     V_mean = V.sum(axis=1)[:, np.newaxis, :]  # Kx1xL
    #
    #     # calculate true gradient squared norm
    #     dRdX = V - R[:, :, np.newaxis] * V_mean  # KxNxL
    #     dFdX = np.einsum("knl,nd->kdl", dRdX, self.Y)  # KxDxL
    #     return dFdX
    #
    # def calc_gradient_sqnorm(self, Xnew):
    #     # calculate gradient squared norm of Xnew
    #     dFdX = self.calc_gradient(Xnew)
    #     return np.sum(dFdX ** 2, axis=(1, 2))  # K
