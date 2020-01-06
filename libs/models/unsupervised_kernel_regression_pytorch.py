# coding: utf-8
import sys
sys.path.append('../')
#from som.datasets.kura import create_data
# from KSE.lib.datasets.artificial.spiral import create_data

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

class Unsupervised_Kernel_Regression_pytorch(object):
    def __init__(self, X, nb_components, bandwidth_gaussian_kernel=1.0,
                 is_compact=False, lambda_=1.0,
                 init='random', is_loocv=False, is_save_history=False):
        self.X = X.clone()
        self.nb_samples = X.shape[0]
        self.nb_dimensions = X.shape[1]
        self.nb_components = nb_components
        self.bandwidth_gaussian_kernel = bandwidth_gaussian_kernel
        self.precision = 1.0 / (bandwidth_gaussian_kernel * bandwidth_gaussian_kernel)
        self.is_compact = is_compact
        self.is_loocv = is_loocv
        self.is_save_hisotry = is_save_history

        self.Z = None
        if isinstance(init, str) and init in 'random':
            self.Z = np.random.normal(0, 1.0, (self.nb_samples, self.nb_components)) * bandwidth_gaussian_kernel * 0.5
        elif isinstance(init, torch.Tensor) and init.shape == (self.nb_samples, self.nb_components):
            self.Z = init
        else:
            raise ValueError("invalid init: {}".format(init))

        self.lambda_ = lambda_

        self._done_fit = False

    def fit(self, nb_epoch=100, verbose=True, eta=0.5, expand_epoch=None):
        self.nb_epoch = nb_epoch

        if self.is_save_hisotry:
            self.history = {}
            self.history['z'] = torch.zeros((nb_epoch, self.nb_samples, self.nb_components))
            self.history['y'] = torch.zeros((nb_epoch, self.nb_samples, self.nb_dimensions))
            self.history['zvar'] = torch.zeros((nb_epoch, self.nb_components))
            self.history['obj_func'] = torch.zeros(nb_epoch)

        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)

        for epoch in bar:
            DistZ = torch.sum((self.Z[:, None, :] - self.Z[None, :, :])**2, dim=2)
            H = torch.exp(-0.5 * self.precision * DistZ)
            if self.is_loocv:
                # H -= np.identity(H.shape[0])
                assert 'Not Implemented Error'
            G = torch.sum(H, dim=1, keepdim=True)
            GInv = 1 / G
            R = H * GInv

            Y = (R @ self.X).clone().detach().requires_grad_(True)
            Error = Y - self.X
            obj_func = torch.sum(Error**2) / self.nb_samples + self.lambda_ * torch.sum(self.Z**2)
            obj_func.backward()
            with torch.no_grad():
                self.Z = self.Z - eta * self.Z.grad
                if self.is_compact:
                    self.Z = torch.clamp(self.Z, -1.0, 1.0)
                else:
                    self.Z = self.Z - self.Z.mean(0)
            self.Z.requires_grad = True

            if self.is_save_hisotry:
                self.history['z'][epoch] = self.Z
                self.history['y'][epoch] = Y
                # self.history['zvar'][epoch] = np.mean(np.square(self.Z - self.Z.mean(axis=0)), axis=0)
                self.history['obj_func'][epoch] = obj_func

        self._done_fit = True
        return self.history
