import numpy as np

from lib.datasets.artificial import sin
from lib.graphics.KSEViewer import KSEViewer
from lib.models.flab.KSE import KSE


def _main():
    np.random.seed(100)
    X = sin.create_data(100)
    X += np.random.normal(0.0,0.3,(X.shape))
    latent_dim = 1
    init = np.random.normal(0, 0.01, (X.shape[0], latent_dim))

    #kse0331 = KSE("0331", X, latent_dim=latent_dim, init=init)
    #kse_standard = KSE("standard", X, latent_dim=latent_dim, init=init)
    kse_0428 = KSE("0428", X, latent_dim=latent_dim, init=init)
    kse_0524 = KSE("0524", X, latent_dim=latent_dim, init=init)

    nb_epoch = 10000
    #kse0331.fit(nb_epoch = nb_epoch)
    #kse_standard.fit(nb_epoch = nb_epoch)
    kse_0428.fit(nb_epoch = nb_epoch)
    kse_0524.fit(nb_epoch = nb_epoch)

    viewer = KSEViewer(kse_0428, rows=2, cols=2, figsize=(12, 3),skip=100)
    # viewer.add_observation_space(kse=kse0331, row=1, col=1, aspect='equal')
    # viewer.add_sequential_space(kse=kse0331,subject_name_list=['gamma', 'beta'], row=2, col=1)
    # viewer.add_observation_space(kse=kse_standard,row=1, col=2, aspect='equal')
    # viewer.add_sequential_space(kse=kse_standard,subject_name_list=['gamma', 'beta'], row=2, col=2)
    viewer.add_observation_space(kse=kse_0428, row=1, col=1, aspect='equal')
    viewer.add_sequential_space(kse=kse_0428,subject_name_list=['gamma', 'beta'], row=2, col=1)
    viewer.add_observation_space(kse=kse_0524,row=1, col=2, aspect='equal')
    viewer.add_sequential_space(kse=kse_0524,subject_name_list=['gamma', 'beta'], row=2, col=2)
    viewer.draw()


if __name__ == "__main__":
    _main()
