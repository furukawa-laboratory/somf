import numpy as np


def create_data(nb_samples, input_dim=2, retdim=False, amplitude=1.0, freq=1.0, phase=0.0):
    latent_dim = 1

    z = np.random.rand(nb_samples, latent_dim) * 2 * np.pi - np.pi

    x = np.zeros((nb_samples, input_dim))
    x[:, 0] = z[:, 0]
    x[:, 1] = amplitude * np.sin(freq * z[:, 0] + phase)

    if retdim:
        return x, latent_dim
    else:
        return x
