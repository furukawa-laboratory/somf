import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import sys
sys.path.append('../../')

from libs.models.som import SOM
from libs.datasets.artificial import animal
from libs.visualization.som.somg import SOMg


if __name__ == '__main__':
    nb_epoch = 50
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.3
    tau = 50
    latent_dim = 2
    seed = 1

    title_text= "animal map"
    umat_resolution = 100 # U-matrix表示の解像度

    X, labels = animal.load_data()

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau)
    som.fit(nb_epoch=nb_epoch)

    somg = SOMg(som)

    # 学習過程の表示
    fig, ax = plt.subplots()
    def unit_distance(k1, k2):
        return np.sum(np.square(som.Y[k1] - som.Y[k2]))
    somg.plot_umatrix(ax, unit_distance)
    plt.show()

    # 学習過程の表示
    fig, ax = plt.subplots()
    def update(frame):
        if frame != 0:
            ax.cla()
        def calc_distance(k1, k2):
            return np.sum(np.square(som.history['y'][frame][k1] - som.history['y'][frame][k2]))
        somg.plot_umatrix(ax, calc_distance)
        ax.set_title(f"t:{frame:03}")

    ani = animation.FuncAnimation(fig, update, frames=nb_epoch, interval=100)
    plt.show()
    # ani.save("somg.gif", writer='imagemagick')
