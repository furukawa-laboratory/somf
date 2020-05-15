#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D

from libs.models.som2 import SOM2

if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    nb_epoch = 100
    n_class = 3
    n_sample = 300
    Dim = 3
    parent_latent_dim = 1
    child_latent_dim = 2
    parent_resolution = 5
    child_resolution = 10
    pCluster_num = parent_resolution ** parent_latent_dim
    cCluster_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.5
    child_sigma_max = 2.0
    child_sigma_min = 0.3
    parent_tau = nb_epoch
    child_tau = nb_epoch
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
    theta = np.linspace(-np.pi / 6, np.pi / 6, n_class)
    for n in range(n_class):
        min_X, max_X = 0, 4
        min_Y, max_Y = -1, 1
        X = np.random.uniform(min_X, max_X, n_sample)
        Y = np.zeros(n_sample)
        for s in range(n_sample):
            deltaY = (max_Y * X[s]) / max_X
            Y[s] = np.random.uniform(-deltaY, deltaY)
        rotate_X = X * np.cos(theta[n]) + Y * np.sin(theta[n])
        rotate_Y = X * np.sin(theta[n]) - Y * np.cos(theta[n])
        rotate_X -= np.mean(rotate_X)
        rotate_Y -= np.mean(rotate_Y)
        Datasets[n][:, 0] = rotate_X
        Datasets[n][:, 1] = rotate_Y
        Datasets[n][:, 2] = n - n_class / 2

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
    ax1 = fig.add_subplot(gs_1[:, :], projection='3d')
    ax2 = fig.add_subplot(gs_2[:, :])

    randomint = np.random.randint(0, child_resolution**child_latent_dim, 5)

    def update(epoch):
        ax1.cla()
        ax2.cla()

        for n in range(n_class):
            ax1.scatter(Datasets[n, :, 0], Datasets[n, :, 1], Datasets[n, :, 2], c=Datasets[n, :, 0],
                        cmap="viridis", marker="+", label='observation data')

            if parent_latent_dim == 2:
                ax2.scatter(pZeta[:, 0], pZeta[:, 1], s=100, c='white', edgecolors='grey', label='Zeta', zorder=1)
                ax2.scatter(pZ[epoch, n, 0], pZ[epoch, n, 1], label='latent variable: Z', zorder=2)
            else:
                ax2.scatter(pZeta, [0] * len(pZeta), s=100, c='white', edgecolors='grey', label='pZeta', zorder=1)
                ax2.scatter(pZ[epoch, n], 0, label='Z', zorder=2)


        for k in range(pCluster_num):
            py = pY[epoch, k].reshape(child_resolution, child_resolution, Dim)
            ax1.plot_wireframe(py[:, :, 0], py[:, :, 1], py[:, :, 2], color='r')

        # unique_bmu = np.unique(bmu)

        # fiberの表示
        # for i in randomint:
        #     ax1.plot(pY[epoch, :, i, 0], pY[epoch, :, i, 1], pY[epoch, :, i, 2], color='r')

        ax1.set_title("observation space", fontsize=9)
        ax2.set_title("latent space(parent)", fontsize=9)
        fig.suptitle("epoch {}/{}".format((epoch + 1), nb_epoch), fontsize=10)
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)

    ani = anim.FuncAnimation(fig, update, interval=interval, frames=nb_epoch, repeat=False)
    ani.save("SOM2.gif", writer='pillow')
    # plt.show()
