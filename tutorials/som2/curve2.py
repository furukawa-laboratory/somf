#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from libs.models.som2 import SOM2

if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    nb_epoch = 20
    n_class = 6
    n_sample = 400
    Dim = 2
    parent_latent_dim = 2
    child_latent_dim = 1
    parent_resolution = 8
    child_resolution = 100
    pCluster_num = parent_resolution ** parent_latent_dim
    cCluster_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.2
    child_sigma_max = 2.0
    child_sigma_min = 0.1
    parent_tau = nb_epoch
    child_tau = nb_epoch
    interval = 500

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
    y = [x[0] ** 3 - 0.5 * x[0], -x[1] ** 3 + 0.5 * x[1], x[2] ** 3 - 0.5 * x[2], -x[3] ** 3 + 0.5 * x[3],
         x[4] ** 3 - 0.5 * x[4], -x[5] ** 3 + 0.5 * x[5]]
    t = 0
    for i in [+1, +1, 0, 0, -1, -1]:
        Datasets[t, :, 0] = x[t] + i
        Datasets[t, :, 1] = y[t] + i
        t += 1

    model = SOM2(Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZ, cZ)
    model.fit(nb_epoch)

    cY = model.history["cY"]
    pY = model.history["pY"]
    cZ = model.history["cZ"]
    pZ = model.history["pZ"]
    pZeta = model.history["pZeta"]
    bmm = model.history["bmm"]

    fig, axes = plt.subplots(parent_resolution, parent_resolution, figsize=(6, 6))

    def update(epoch):
        for i in range(parent_resolution):
            for j in range(parent_resolution):
                axes[i, j].cla()

        unique_bmm = np.unique(bmm[epoch])
        list_bmm = [0]*pCluster_num
        for i in unique_bmm:
            list_bmm[int(i)] = 1

        t = 0
        for i in range(parent_resolution):
            for j in range(parent_resolution):
                if list_bmm[t] == 1:
                    axes[i, j].scatter(pY[epoch, t, :, 0], pY[epoch, t, :, 1], s=5, color='r', label='bmm')
                    axes[i, j].set_facecolor('k')
                    # axes[i, j].legend(loc=2, prop={'size': 5})
                else:
                    axes[i, j].scatter(pY[epoch, t, :, 0], pY[epoch, t, :, 1], s=5, color='grey')
                    axes[i, j].set_facecolor('w')
                t += 1
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].set_xlim(-2, 2)
                axes[i, j].set_ylim(-2, 2)

        fig.suptitle("epoch {}/{} latent space(parent)".format((epoch + 1), nb_epoch), fontsize=10)

    ani = anim.FuncAnimation(fig, update, interval=interval, frames=nb_epoch, repeat=False)
    # ani.save("SOM2.gif", writer='pillow')
    plt.show()
