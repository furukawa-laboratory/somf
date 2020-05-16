#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D

from libs.models.som2 import SOM2

if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    nb_epoch = 100
    n_class = 6
    n_sample = 400
    Dim = 2
    parent_latent_dim = 2
    child_latent_dim = 1
    parent_resolution = 4
    child_resolution = 100
    pCluster_num = parent_resolution ** parent_latent_dim
    cCluster_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.5
    child_sigma_max = 2.0
    child_sigma_min = 0.2
    parent_tau = nb_epoch * 0.8
    child_tau = nb_epoch * 0.8
    interval = 100

    # ZetaとZの初期値を生成
    if parent_latent_dim == 2:
        pZ = np.random.normal(size=(n_class, parent_latent_dim), loc=0, scale=0.01)
    else:
        pZ = np.random.normal(size=(n_class, 1), loc=0.0, scale=0.01)

    if child_latent_dim == 2:
        cZ = np.random.normal(size=(n_sample, child_latent_dim), loc=0, scale=0.01)
    else:
        cZ = np.random.normal(size=(n_sample, 1), loc=0.0, scale=0.01)

    # データ生成
    Datasets = np.zeros((n_class, n_sample, Dim))
    x = np.random.rand(n_class, n_sample) * 2 - 1
    x = np.sort(x)
    y = [0.5*x[0], -0.5*x[1], x[2] ** 2, -(x[3] ** 2), x[4] ** 3 - 0.5 * x[4], -x[5] ** 3 + 0.5 * x[5]]
    for n, (a, b) in enumerate(zip(x, y)):
        Datasets[n, :, 0] = a
        Datasets[n, :, 1] = b

    model = SOM2(Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZ, cZ)
    model.fit(nb_epoch)

    cY = model.history["cY"]
    pY = model.history["pY"]
    cZ = model.history["cZ"]
    pZ = model.history["pZ"]
    pZeta = model.history["pZeta"]
    bmu = model.history["bmu"]

    # 描画
    fig = plt.figure(figsize=(10, 5))
    gs_master = GridSpec(nrows=1, ncols=2)
    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[:, 0:1])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[:, 1:2])
    ax1 = fig.add_subplot(gs_1[:, :])
    ax2 = fig.add_subplot(gs_2[:, :])

    randomint = np.random.randint(0, child_resolution**child_latent_dim, 8)

    def update(epoch):
        ax1.cla()
        ax2.cla()

        for n in range(n_class):
            ax1.scatter(Datasets[n, :, 0], Datasets[n, :, 1], c=Datasets[n, :, 0], s=20, label='observation data')

            if parent_latent_dim == 2:
                ax2.scatter(pZeta[:, 0], pZeta[:, 1], s=100, c='white', edgecolors='grey', label='Zeta', zorder=1)
                ax2.scatter(pZ[epoch, n, 0], pZ[epoch, n, 1], label='latent variable: Z', zorder=2)

        for k in range(pCluster_num):
            ax1.plot(pY[epoch, k, :, 0], pY[epoch, k, :, 1], color='r')

        # unique_bmu = np.unique(bmu)

        # fiberの表示
        # for i in randomint:
        #     ax1.plot(pY[epoch, :, i, 0], pY[epoch, :, i, 1], color='b')

        ax1.set_title("observation space", fontsize=9)
        ax2.set_title("latent space(parent)", fontsize=9)
        fig.suptitle("epoch {}/{}".format((epoch + 1), nb_epoch), fontsize=10)
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)

    ani = anim.FuncAnimation(fig, update, interval=interval, frames=nb_epoch, repeat=False)
    # ani.save("SOM2.gif", writer='pillow')
    plt.show()
