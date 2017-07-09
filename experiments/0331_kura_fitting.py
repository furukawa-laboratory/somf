from lib.datasets.artificial import kura
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.models.KSE import KSE


def _main():
    version = "0331"
    X = kura.create_data(100)
    latent_dim = 2
    init = 'random'

    kse = KSE(version, X, latent_dim=latent_dim, init=init)
    kse.fit()
    Y = kse.history['y'][-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], label='X')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], label='Y', c='r')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _main()
