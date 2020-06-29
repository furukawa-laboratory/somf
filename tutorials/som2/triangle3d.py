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
    n_sample_list = np.random.randint(low=100, high=150, size=n_class)
    Dim = 3
    parent_latent_dim = 1
    child_latent_dim = 2
    parent_resolution = 5
    child_resolution = 10
    parent_node_num = parent_resolution ** parent_latent_dim
    child_node_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.5
    child_sigma_max = 2.0
    child_sigma_min = 0.2
    parent_tau = nb_epoch
    child_tau = nb_epoch
    interval = 100

    assert parent_latent_dim in [1, 2], "parent_latent_dim must be 1 or 2."
    assert child_latent_dim in [1, 2], "child_latent_dim must be 1 or 2."


    # データ生成
    datasets = []
    theta = np.linspace(-np.pi / 12, np.pi / 12, n_class)
    for n in range(n_class):
        dataset = []
        min_X, max_X = 0, 4
        min_Y, max_Y = -1, 1
        X = np.random.uniform(min_X, max_X, n_sample_list[n])
        Y = np.zeros(n_sample_list[n])
        for s in range(n_sample_list[n]):
            deltaY = (max_Y * X[s]) / max_X
            Y[s] = np.random.uniform(-deltaY, deltaY)
        rotate_X = X * np.cos(theta[n]) + Y * np.sin(theta[n])
        rotate_Y = X * np.sin(theta[n]) - Y * np.cos(theta[n])
        rotate_X -= np.mean(rotate_X)
        rotate_Y -= np.mean(rotate_Y)
        stack = np.dstack([rotate_X, rotate_Y, [n - n_class / 2] * n_sample_list[n]])
        datasets.append(stack.tolist()[0])


    params_1st_som = {
        "latent_dim": child_latent_dim,
        "resolution": child_resolution,
        "sigma_max": child_sigma_max,
        "sigma_min": child_sigma_min,
        "tau": child_tau,
        "init": "random",
    }

    params_2nd_som = {
        "latent_dim": parent_latent_dim,
        "resolution": parent_resolution,
        "sigma_max": parent_sigma_max,
        "sigma_min": parent_sigma_min,
        "tau": parent_tau,
        "init": "random",
    }

    model = SOM2(datasets, params_1st_som, params_2nd_som, is_save_history=True)
    model.fit(nb_epoch)

    cY = model.history["cY"]
    pY = model.history["pY"]
    cZ = model.history["cZ"]
    pZ = model.history["pZ"]
    pZeta = model.history["pZeta"]
    bmu = model.history["bmu"]


    # 描画
    fig = plt.figure(figsize=(8, 4))
    gs_master = GridSpec(nrows=1, ncols=2)
    gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[:, 0:1])
    gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[:, 1:2])
    ax1 = fig.add_subplot(gs_1[:, :], projection='3d')
    ax2 = fig.add_subplot(gs_2[:, :])


    def update(epoch):
        ax1.cla()
        ax2.cla()

        for n in range(n_class):
            if isinstance(datasets[n], list):
                data = np.array(datasets[n])
                ax1.scatter(data[:, 0], data[:, 1], data[:, 2],
                            marker="+", label='observation data')
            else:
                ax1.scatter(datasets[n][:, 0], datasets[n][:, 1], datasets[n][:, 2],
                            marker="+", label='observation data')

            if parent_latent_dim == 2:
                ax2.scatter(pZeta[:, 0], pZeta[:, 1], s=100, c='white', edgecolors='grey', label='Zeta', zorder=1)
                ax2.scatter(pZ[epoch, n, 0], pZ[epoch, n, 1], label='latent variable: Z', zorder=2)
            else:
                ax2.scatter(pZeta, [0] * len(pZeta), s=100, c='white', edgecolors='grey', label='pZeta', zorder=1)
                ax2.scatter(pZ[epoch, n], 0, label='Z', zorder=2)


        for k in range(parent_node_num):
            if child_latent_dim == 2:
                py = pY[epoch, k].reshape(child_resolution, child_resolution, Dim)
                ax1.plot_wireframe(py[:, :, 0], py[:, :, 1], py[:, :, 2], color='r')
            else:
                ax1.plot(pY[epoch, k, :, 0], pY[epoch, k, :, 1], pY[epoch, k, :, 2], color='r')

        # fiberの表示
        for i in [0, child_resolution-1, child_node_num-child_resolution, child_node_num-1]:
            ax1.plot(pY[epoch, :, i, 0], pY[epoch, :, i, 1], pY[epoch, :, i, 2], color='b')

        ax1.set_title("observation space", fontsize=9)
        ax2.set_title("latent space(parent)", fontsize=9)
        fig.suptitle("epoch {}/{}".format((epoch + 1), nb_epoch), fontsize=10)
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)

    ani = anim.FuncAnimation(fig, update, interval=interval, frames=nb_epoch, repeat=False)
    # ani.save("SOM2.gif", writer='pillow')
    plt.show()
