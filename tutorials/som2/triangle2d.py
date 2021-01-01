#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from libs.models.som2 import SOM2

if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    nb_epoch = 25
    n_class = 9
    n_sample_list = np.random.randint(low=50, high=200, size=n_class)
    Dim = 2
    parent_latent_dim = 2
    child_latent_dim = 2
    parent_resolution = 9
    child_resolution = 10
    parent_node_num = parent_resolution ** parent_latent_dim
    child_node_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.3
    child_sigma_max = 2.0
    child_sigma_min = 0.2
    parent_tau = nb_epoch
    child_tau = nb_epoch
    interval = 200

    assert n_class == 9, "n_class must be 9."

    # データ生成
    datasets = []
    theta = [0,          np.pi/6,    np.pi/3,
             -np.pi/6,         0,    np.pi/6,
             -np.pi/3,  -np.pi/6,          0]
    postion_x = [-1, 0, 1, -1, 0, 1, -1,  0,  1]
    postion_y = [ 1, 1, 1,  0, 0, 0, -1, -1, -1]
    for n, n_sample in enumerate(n_sample_list):
        min_X, max_X = 0, 3
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
        x = rotate_X + postion_x[n] * 2
        y = rotate_Y + postion_y[n] * 2
        datasets.append(np.dstack([x, y]).tolist()[0])

    # Zの初期値を生成
    init_bmus = []
    for n_sample in n_sample_list:
        init_bmus.append(np.random.randint(low=0, high=child_node_num, size=n_sample, dtype="int"))
    init_bmm = np.random.randint(low=0, high=parent_node_num, size=n_class, dtype="int")

    params_1st_som = {
        "latent_dim": child_latent_dim,
        "resolution": child_resolution,
        "sigma_max": child_sigma_max,
        "sigma_min": child_sigma_min,
        "tau": child_tau,
    }

    params_2nd_som = {
        "latent_dim": parent_latent_dim,
        "resolution": parent_resolution,
        "sigma_max": parent_sigma_max,
        "sigma_min": parent_sigma_min,
        "tau": parent_tau,
    }

    model = SOM2(datasets, params_1st_som, params_2nd_som, init_bmus, init_bmm, is_save_history=True)
    model.fit(nb_epoch)

    cY = model.history["cY"]
    pY = model.history["pY"]
    cZ = model.history["cZ"]
    pZ = model.history["pZ"]
    pZeta = model.history["pZeta"]
    bmm = model.history["bmm"]

    fig, axes = plt.subplots(parent_resolution, parent_resolution, figsize=(7, 5))

    def update(epoch):
        for i in range(parent_resolution):
            for j in range(parent_resolution):
                axes[i, j].cla()

        unique_bmm = np.unique(bmm[epoch])
        list_bmm = [0]*parent_node_num
        for i in unique_bmm:
            list_bmm[int(i)] = 1

        t = 0
        for i in range(parent_resolution):
            for j in range(parent_resolution):
                if list_bmm[t] == 1:
                    axes[i, j].scatter(pY[epoch, t, :, 0], pY[epoch, t, :, 1], s=5)
                    axes[i, j].set_facecolor('k')
                else:
                    axes[i, j].scatter(pY[epoch, t, :, 0], pY[epoch, t, :, 1], s=5, color='grey')
                    axes[i, j].set_facecolor('w')
                t += 1
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        fig.suptitle("epoch {}/{} latent space(parent)".format((epoch + 1), nb_epoch), fontsize=10)

    ani = anim.FuncAnimation(fig, update, interval=interval, frames=nb_epoch, repeat=False)
    # ani.save("SOM2.gif", writer='pillow')
    plt.show()

