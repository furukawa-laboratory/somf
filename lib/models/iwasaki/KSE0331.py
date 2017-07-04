import numpy as np


class KSE(object):
    def __init__(self, X, latent_dim, init='random'):
        self.X = X.copy()
        self.nb_samples = X.shape[0]
        self.input_dim = X.shape[1]
        self.latent_dim = latent_dim

        self.Z = init

        self.history = {}

    def fit(self, nb_epoch, epsilon=0.5):

        return self.history