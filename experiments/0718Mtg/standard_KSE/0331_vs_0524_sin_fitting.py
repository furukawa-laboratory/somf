from lib.datasets.artificial import sin
from lib.models.flab.KSE import KSE
from lib.graphics.KSEViewer import KSEViewer
import numpy as np


def _main():
    np.random.seed(100)
    X = sin.create_data(100)
    latent_dim = 1
    init = np.random.normal(0, 0.01, (X.shape[0], latent_dim))

    kse0331 = KSE("0331", X, latent_dim=latent_dim, init=init)
    kse0524 = KSE("0524", X, latent_dim=latent_dim, init=init)

    nb_epoch = 1000
    kse0331.fit(nb_epoch = nb_epoch)
    kse0524.fit(nb_epoch = nb_epoch)

    viewer = KSEViewer(kse0331, rows=2, cols=2, figsize=(10, 5),skip=10)
    viewer.add_observation_space(kse=kse0331, row=1, col=1, aspect='equal')
    viewer.add_sequential_space(kse=kse0331,subject_name_list=['gamma', 'beta'], row=2, col=1)
    viewer.add_observation_space(kse=kse0524,row=1, col=2, aspect='equal')
    viewer.add_sequential_space(kse=kse0524,subject_name_list=['gamma', 'beta'], row=2, col=2)
    viewer.draw()
    viewer.save_gif('0331_vs_0524.gif')


if __name__ == "__main__":
    _main()
