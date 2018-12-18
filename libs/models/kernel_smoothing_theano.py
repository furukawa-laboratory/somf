# -*- coding: utf-8 -*-
import numpy as np
from libs.models import KernelSmoothing
import theano
import theano.tensor as tt


class KernelSmoothingTheano(KernelSmoothing):
    def fit(self, X, Y):
        super(KernelSmoothingTheano, self).fit(X, Y)
        self.X = theano.shared(self.X.copy())
        self.Y = theano.shared(self.Y.copy())

    def _difine_ks(self, flag='map'):
        xnew = tt.dmatrix('xnew')
        xnew1 = xnew.dimshuffle(0, 'x', 1)
        Delta = xnew1 - self.X[np.newaxis, :, :]
        Dist = tt.sum(tt.square(Delta), axis=2)
        H = tt.exp(-0.5 * Dist / (self.sigma * self.sigma))
        G = tt.sum(H, axis=1).dimshuffle(0, 'x')
        GInv = 1.0 / G
        R = H * GInv
        F = tt.dot(R, self.Y)
        Grad = tt.grad(tt.sum(F), xnew)

        if flag == 'map':
            func = theano.function(inputs=[xnew], outputs=[F])
            return func
        elif flag == 'grad':
            func = theano.function(inputs=[xnew], outputs=[Grad])
            return func

    def predict(self, Xnew):
        # check format of Xnew
        self._check_correct_ndarray(Xnew, "Xnew")
        Xnew = Xnew.reshape(Xnew.shape[0], -1)
        if Xnew.shape[1] != self.input_dim:
            raise ValueError("X and Xnew must be same dimension")

        func = self._difine_ks('map')

        fnew, = func(Xnew)
        return fnew

    def calc_gradient(self, Xnew):
        # check format of Xnew
        self._check_correct_ndarray(Xnew, "Xnew")
        Xnew = Xnew.reshape(Xnew.shape[0], -1)
        if Xnew.shape[1] != self.input_dim:
            raise ValueError("X and Xnew must be same dimension")

        if self.Y.get_value().shape[1] != 1:
            raise ValueError("not support 2 or more dimension of Y")

        func = self._difine_ks('grad')
        grad_new, = func(Xnew)
        return grad_new

    def calc_gradient_sqnorm(self, Xnew):
        # calculate gradient squared norm of Xnew
        dFdXnew = self.calc_gradient(Xnew)
        return np.sum(dFdXnew ** 2, axis=1)  # K
