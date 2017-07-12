from lib.datasets.artificial import kura
from lib.models.KSE import KSE
from lib.graphics.KSEViewer import KSEViewer


def _main():
    version = "0331"
    X = kura.create_data(100)
    latent_dim = 2
    init = 'random'

    kse = KSE(version, X, latent_dim=latent_dim, init=init)
    kse.fit()

    viewer = KSEViewer(kse, rows=2, cols=2)
    viewer.add_observation_space(row=2, col=1, aspect='equal')
    viewer.draw()


if __name__ == "__main__":
    _main()
