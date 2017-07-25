import numpy as np

from lib.datasets.artificial import sin
from lib.graphics.KSEViewer import KSEViewer
from lib.models.flab.KSE import KSE


def _main():
    input_dim = 2
    np.random.seed(100)
    X = sin.create_data(100, input_dim=input_dim)
    X += np.random.normal(0.0, 0.3, X.shape) / input_dim
    latent_dim = 1
    init = np.random.normal(0, 0.01, (X.shape[0], latent_dim))

    kse0524 = KSE("0524", X, latent_dim=latent_dim, init=init)

    nb_epoch = 5000
    kse0524.fit(nb_epoch = nb_epoch, gamma_divisor=input_dim)

    viewer = KSEViewer(kse0524, rows=2, cols=1, figsize=(5, 5),skip=10)
    viewer.add_observation_space(kse=kse0524, row=1, col=1, aspect='equal')
    viewer.add_sequential_space(kse=kse0524, subject_name_list=['gamma', 'beta'], row=2, col=1)
    viewer.draw()


if __name__ == "__main__":
    _main()
