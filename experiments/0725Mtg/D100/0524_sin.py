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

        kse0524_divD = KSE("0524", X, latent_dim=latent_dim, init=init)
        kse0524_div2 = KSE("0524", X, latent_dim=latent_dim, init=init)

        nb_epoch = 5000
        kse0524_divD.fit(nb_epoch = nb_epoch, gamma_divisor=input_dim, gamma_update_freq=5)
        kse0524_div2.fit(nb_epoch = nb_epoch, gamma_divisor=2, gamma_update_freq=5)

        viewer = KSEViewer(kse0524_divD, rows=2, cols=2, figsize=(10, 5),skip=100)
        viewer.add_observation_space(kse=kse0524_divD, row=1, col=1, aspect='equal', title=r'$\gamma = \frac{1}{D} a \beta$ ' + '$(D={})$'.format(input_dim))
        viewer.add_sequential_space(kse=kse0524_divD, subject_name_list=['gamma', 'beta'], row=2, col=1)
        viewer.add_observation_space(kse=kse0524_div2, row=1, col=2, aspect='equal', title=r'$\gamma = \frac{1}{2} a \beta$')
        viewer.add_sequential_space(kse=kse0524_div2, subject_name_list=['gamma', 'beta'], row=2, col=2)
        viewer.draw()
        viewer.save_png(filename='524_D_{}.png'.format(input_dim))

if __name__ == "__main__":
    _main()
