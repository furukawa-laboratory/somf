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
    interval = 100

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

    fig, axes = plt.subplots(parent_resolution, parent_resolution, figsize=(6, 6))

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
                    axes[i, j].scatter(pY[epoch, t, :, 0], pY[epoch, t, :, 1], s=5, color='r')
                    axes[i, j].set_facecolor('k')
                    # axes[i, j].legend(loc=2, prop={'size': 5})
                else:
                    axes[i, j].scatter(pY[epoch, t, :, 0], pY[epoch, t, :, 1], s=5, color='grey')
                    axes[i, j].set_facecolor('w')
                t += 1
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                # axes[i, j].set_xlim(-1, 1)
                # axes[i, j].set_ylim(-1, 1)

        fig.suptitle("epoch {}/{} latent space(parent)".format((epoch + 1), nb_epoch), fontsize=10)

    # 描画
    # fig = plt.figure(figsize=(8, 4))
    # gs_master = GridSpec(nrows=1, ncols=2)
    # gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[:, 0:1])
    # gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[:, 1:2])
    # ax1 = fig.add_subplot(gs_1[:, :])
    # ax2 = fig.add_subplot(gs_2[:, :])

    # def update(epoch):
    #     ax1.cla()
    #     ax2.cla()
    #
    #     for n in range(n_class):
    #         if isinstance(datasets[n], list):
    #             data = np.array(datasets[n])
    #             ax1.scatter(data[:, 0], data[:, 1],
    #                         marker="+", label='observation data')
    #         else:
    #             ax1.scatter(datasets[n][:, 0], datasets[n][:, 1],
    #                         marker="+", label='observation data')
    #
    #         if parent_latent_dim == 2:
    #             ax2.scatter(pZeta[:, 0], pZeta[:, 1], s=100, c='white', edgecolors='grey', label='Zeta', zorder=1)
    #             ax2.scatter(pZ[epoch, n, 0], pZ[epoch, n, 1], label='latent variable: Z', zorder=2)
    #         else:
    #             ax2.scatter(pZeta, [0] * len(pZeta), s=100, c='white', edgecolors='grey', label='pZeta', zorder=1)
    #             ax2.scatter(pZ[epoch, n], 0, label='Z', zorder=2)
    #
    #
    #     for k in range(parent_node_num):
    #         if child_latent_dim == 2:
    #             py = pY[epoch, k].reshape(child_resolution, child_resolution, Dim)
    #             for r in range(child_resolution):
    #                 ax1.plot(py[r, :, 0], py[r, :, 1], color='r', linewidth=1)
    #                 ax1.plot(py[:, r, 0], py[:, r, 1], color='r', linewidth=1, zorder=0)
    #         else:
    #             ax1.plot(pY[epoch, k, :, 0], pY[epoch, k, :, 1], color='r')
    #
    #     # fiberの表示
    #     color_li = ['#A5BEFA', '#B3093F', "#451531", "#64B7CC"]
    #     for i, node in enumerate([0, child_resolution-1, child_node_num-child_resolution, child_node_num-1]):
    #         ax1.scatter(pY[epoch, :, node, 0], pY[epoch, :, node, 1], s=20, color=color_li[i], zorder=1)
    #
    #     ax1.set_title("observation space", fontsize=9)
    #     ax2.set_title("latent space(parent)", fontsize=9)
    #     fig.suptitle("epoch {}/{}".format((epoch + 1), nb_epoch), fontsize=10)
    #     ax2.set_xlim(-1.2, 1.2)
    #     ax2.set_ylim(-1.2, 1.2)

    ani = anim.FuncAnimation(fig, update, interval=interval, frames=nb_epoch, repeat=False)
    # ani.save("SOM2.gif", writer='pillow')
    plt.show()

