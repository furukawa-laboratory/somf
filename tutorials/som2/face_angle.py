#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from PIL import Image

from libs.models.som2 import SOM2

if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    nb_epoch = 20
    max_angle = 80
    pixel_size = 64
    n_class = 90
    n_sample = int(2 * max_angle / 5 + 1)
    Dim = pixel_size ** 2
    parent_latent_dim = 2
    child_latent_dim = 1
    parent_resolution = 10
    child_resolution = 100
    pCluster_num = parent_resolution ** parent_latent_dim
    cCluster_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.3
    child_sigma_max = 2.0
    child_sigma_min = 0.2
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

    # マルチアングル画像は古川研のdropbox上にあります
    Datasets = np.zeros((n_class, n_sample, Dim))
    Dir_resized = "../../../Angle_resized/"
    angles = np.linspace(-max_angle, max_angle, n_sample, dtype="int8")
    for i in range(n_class):
        for n, angle in enumerate(angles):
            Sbj = "{}/A_{:02d}_".format(angle, i + 1)
            if angle == 0:
                file_name = "0.jpg"
            else:
                file_name = "{0:+03d}.jpg".format(angle)
            img = (
                Image.open(Dir_resized + Sbj + file_name)
                .resize((pixel_size, pixel_size))
                .convert("L")
            )
            Datasets[i, n, :] = np.reshape(img, pixel_size ** 2)

    model = SOM2(
        Datasets,
        parent_latent_dim,
        child_latent_dim,
        parent_resolution,
        child_resolution,
        parent_sigma_max,
        child_sigma_max,
        parent_sigma_min,
        child_sigma_min,
        parent_tau,
        child_tau,
        pZ,
        cZ,
    )
    model.fit(nb_epoch)

    cY = model.history["cY"]
    pY = model.history["pY"]
    cZ = model.history["cZ"]
    pZ = model.history["pZ"]
    pZeta = model.history["pZeta"]
    bmu = model.history["bmu"]

    fig, axes = plt.subplots(parent_resolution, parent_resolution, figsize=(7, 7))

    def update(epoch):
        for i in range(parent_resolution):
            for j in range(parent_resolution):
                axes[i, j].cla()

        unique_bmu = np.unique(bmu[epoch])
        list_bmu = [0] * pCluster_num
        for i in unique_bmu:
            list_bmu[int(i)] = 1

        t = 0
        for i in range(parent_resolution):
            for j in range(parent_resolution):
                axes[i, j].imshow(
                    pY[epoch, t, 0].reshape(pixel_size, pixel_size), "gray"
                )
                t += 1
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        fig.suptitle(
            "epoch {}/{} latent space(parent)".format((epoch + 1), nb_epoch),
            fontsize=10,
        )

    ani = anim.FuncAnimation(
        fig, update, interval=interval, frames=nb_epoch, repeat=False
    )
    ani.save("SOM2.gif", writer='pillow')
    # plt.show()
