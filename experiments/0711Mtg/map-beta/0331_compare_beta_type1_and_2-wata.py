from lib.datasets.artificial import sin
from lib.models.watanabe.KSE0331_add_changing_beta import KSE
from lib.graphics.KSEViewer import KSEViewer
import numpy as np


def _main():
    np.random.seed(122)
    X = sin.create_data(100)
    latent_dim = 1
    init = 'random'

    betaType1 = 'type1'
    betaType2 = 'type2'

    X += np.random.normal(0, 0.1, X.shape)

    nb_epoch = 500

    kse1 = KSE(X, latent_dim=latent_dim, init=init, choice_beta=betaType1)
    kse2 = KSE(X, latent_dim=latent_dim, init=init, choice_beta=betaType2)
    kse1.fit(nb_epoch = nb_epoch)
    kse2.fit(nb_epoch = nb_epoch)

    viewer = KSEViewer(kse1, rows=2, cols=2, figsize=(10, 6),skip=10)
    viewer.add_observation_space(kse=kse1, row=1, col=1, aspect='equal', title=betaType1)
    viewer.add_sequential_space(kse=kse1, subject_name_list=['gamma', 'beta'], row=2, col=1)
    viewer.add_observation_space(kse=kse2, row=1, col=2, aspect='equal', title=betaType2)
    viewer.add_sequential_space(kse=kse2, subject_name_list=['gamma', 'beta'], row=2, col=2)
    viewer.draw()
    #viewer.save_gif('compare_beta_type.gif')

if __name__ == "__main__":
    _main()
