import numpy as np

from lib.datasets.artificial import sin
from lib.graphics.KSEViewer import KSEViewer
from lib.models.flab.KSE import KSE


def _main():
    version = "0331"
    np.random.seed(100)
    X = sin.create_data(100)
    latent_dim = 2
    init = 'random'

    X += np.random.normal(0, 0.1, X.shape)

    kse = KSE(version, X, latent_dim=latent_dim, init=init)
    kse.fit()

    viewer = KSEViewer(kse, rows=2, cols=1, figsize=(6, 6))
    viewer.add_observation_space(row=1, col=1, aspect='equal')
    viewer.add_sequential_space(['gamma', 'beta'], row=2, col=1)
    viewer.draw()


if __name__ == "__main__":
    _main()
