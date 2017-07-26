import numpy as np

from lib.datasets.artificial import sin
from lib.graphics.KSEViewer import KSEViewer
from lib.models.flab.KSE import KSE


def _main():
    input_dims = [2, 3, 5, 10, 100]
    for input_dim in input_dims:
        print(input_dim)
        np.random.seed(100)
        X = sin.create_data(100, input_dim=input_dim)
        X += np.random.normal(0.0, 0.3, X.shape) / input_dim
        latent_dim = 1
        init = np.random.normal(0, 0.01, (X.shape[0], latent_dim))

        kse_331 = KSE("0331", X, latent_dim=latent_dim, init=init)
        kse_standard = KSE("standard", X, latent_dim=latent_dim, init=init)

        nb_epoch = 1000

        lamb = 30**2
        alpha = 1.0 / input_dim
        kse_331.fit(nb_epoch = nb_epoch, alpha = alpha, gamma = lamb)
        kse_standard.fit(nb_epoch = nb_epoch, lamb = lamb)

        print(np.allclose(kse_331.history['z'], kse_standard.history['z'], atol=1e-6))
        print(np.allclose(kse_331.history['beta'], kse_standard.history['beta/D']))

        viewer = KSEViewer(kse_331, rows=2, cols=2, figsize=(10, 5),skip=100)
        viewer.add_observation_space(kse=kse_331, row=1, col=1, aspect='equal', title=r'331 $D={}$'.format(input_dim))
        viewer.add_sequential_space(kse=kse_331, subject_name_list=['beta'], row=2, col=1)
        viewer.add_observation_space(kse=kse_standard, row=1, col=2, aspect='equal', title=r'standard $D={}$'.format(input_dim))
        viewer.add_sequential_space(kse=kse_standard, subject_name_list=['beta','beta/D'], row=2, col=2)
        viewer.draw()
        viewer.save_png(filename='standard_D_{}.png'.format(input_dim))

if __name__ == "__main__":
    _main()
