import numpy as np
import matplotlib.pyplot as plt

from libs.models.som.som import SOM
from libs.datasets.real import oilflow





def _main():
    # input oilflow data
    X, Label = oilflow.load_data()
    nb_samples, visible_dim = X.shape
    latent_dim = 2

    # random_seed
    random_seed = 100
    np.random.seed(random_seed)

    # SOM parameters
    resolution_som = 10
    nb_nodes = resolution_som**latent_dim

    sigma_max = 1.2
    sigma_min = 0.3
    tau = 100
    nb_epoch_som = 100
    init = np.random.rand(nb_samples,latent_dim)*2.0-1.0

    som = SOM(X, latent_dim=latent_dim, resolution=resolution_som,
              sigma_max=sigma_max, sigma_min=sigma_min, tau=tau, init=init)
    som.fit(nb_epoch_som)


    # prepare figure
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111,aspect='equal')

    cmaps = ["b", "r", "g"]
    mmaps = ["<", ">", "^"]

    # plot som
    Z = som.Z

    for n in range(nb_samples):
        ax.scatter(Z[n, 0], Z[n, 1], c=cmaps[Label[n]], marker=mmaps[Label[n]], edgecolors='k')

    plt.show()

    print('finished')


if __name__ == '__main__':
    _main()
