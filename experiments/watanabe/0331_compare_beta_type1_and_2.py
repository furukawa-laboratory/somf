from lib.datasets.artificial import sin
from lib.models.watanabe.KSE0331_add_changing_beta import KSE
from lib.graphics.KSEViewer import KSEViewer
import numpy as np


def _main():
    X = sin.create_data(100)
    latent_dim = 2
    init = 'random'

    np.random.seed(100)
    X += np.random.normal(0, 0.1, X.shape)

    nb_epoch = 500

    kse1 = KSE(X, latent_dim=latent_dim, init=init, choice_beta='type1')
    kse2 = KSE(X, latent_dim=latent_dim, init=init, choice_beta='type2')
    kse1.fit(nb_epoch = nb_epoch)
    kse2.fit(nb_epoch = nb_epoch)

    # viewer1 = KSEViewer(kse1, rows=2, cols=1)
    # viewer1.add_observation_space(row=1, col=1, aspect='equal')
    # viewer1.add_sequential_space(['gamma', 'beta'], row=2, col=1)
    # viewer1.draw()

    viewer2 = KSEViewer(kse2, rows=2, cols=1, size=4)
    viewer2.add_observation_space(row=1, col=1, aspect='equal')
    viewer2.add_sequential_space(['gamma', 'beta'], row=2, col=1)
    viewer2.draw()
    #viewer2.save_gif('result_changing_beta.gif')

if __name__ == "__main__":
    _main()
