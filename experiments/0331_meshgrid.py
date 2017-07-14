from lib.datasets.artificial import sin, kura
from lib.models.iwasaki.KSE0331 import KSE
from lib.graphics.KSEViewer import KSEViewer
import numpy as np


def _main():
    np.random.seed(100)
    kura_data = kura.create_data(100)
    sin_data = sin.create_data(100)

    kse_kura = KSE(kura_data, latent_dim=2, init='random')
    kse_sin = KSE(sin_data, latent_dim=1, init='random')

    kse_kura.fit()
    kse_sin.fit()

    viewer = KSEViewer(kse_kura, rows=2, cols=2, figsize=(10, 6))
    viewer.add_observation_space(kse=kse_kura, row=1, col=1, aspect='equal', projection='3d')
    viewer.add_sequential_space(kse=kse_kura, subject_name_list=['gamma', 'beta'], row=2, col=1)
    viewer.add_observation_space(kse=kse_sin, row=1, col=2, aspect='equal')
    viewer.add_sequential_space(kse=kse_sin, subject_name_list=['gamma', 'beta'], row=2, col=2)
    viewer.draw()


if __name__ == "__main__":
    _main()
