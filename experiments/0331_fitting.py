from lib.datasets.artificial import kura
import matplotlib.pyplot as plt
from lib.models.KSE import KSE


def _main():
    version = "0331"
    X = kura.create_data(100)
    plt.plot(X[:, 0], X[:, 1], '.')
    plt.show()

    kse = KSE(version, X)
    kse.fit()

if __name__ == "__main__":
    _main()
