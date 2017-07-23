from lib.datasets.artificial import kura
from lib.models.KSE import KSE
from lib.graphics.KSEViewer import KSEViewer
import numpy as np


def _main():
    np.random.seed(100)
    X = kura.create_data(100)
    latent_dim = 2
    init = np.random.normal(0, 0.1, (X.shape[0], latent_dim))

    kse0331 = KSE("0331", X, latent_dim=latent_dim, init=init)
    kse_standard = KSE("standard", X, latent_dim=latent_dim, init=init)

    nb_epoch = 500
    kse0331.fit(nb_epoch = nb_epoch)
    kse_standard.fit(nb_epoch = nb_epoch)

    viewer = KSEViewer(kse0331, rows=2, cols=2, figsize=(10, 6),skip=10)
    viewer.add_observation_space(kse=kse0331, row=1, col=1, aspect='equal', projection='3d')
    viewer.add_sequential_space(kse=kse0331,subject_name_list=['gamma', 'beta'], row=2, col=1)
    viewer.add_observation_space(kse=kse_standard,row=1, col=2, aspect='equal', projection='3d')
    viewer.add_sequential_space(kse=kse_standard,subject_name_list=['gamma', 'beta'], row=2, col=2)
    viewer.draw()


if __name__ == "__main__":
    _main()
