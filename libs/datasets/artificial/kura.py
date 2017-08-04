import numpy as np


def create_data(nb_samples, input_dim=3, retdim=False):
    latent_dim = 2

    z1 = np.random.rand(nb_samples) * 2.0 - 1.0
    z2 = np.random.rand(nb_samples) * 2.0 - 1.0

    x = np.zeros((nb_samples, input_dim))
    x[:, 0] = z1
    x[:, 1] = z2
    x[:, 2] = z1 ** 2 - z2 ** 2

    if retdim:
        return x, latent_dim
    else:
        return x
