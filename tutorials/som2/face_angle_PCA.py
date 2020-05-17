#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.decomposition import PCA
from libs.models.som2 import SOM2

if __name__ == "__main__":
    seed = 1
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

    # データ生成
    Datasets = np.zeros((n_class, n_sample, Dim))
    Dir_resized = "../../../Angle_resized/"
    # face_imgs = np.zeros((n_class, pixel_size, pixel_size))
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
            Datasets[i, n, :] = np.reshape(img, pixel_size ** 2) / 255.0

    # Datasetsの平均を0にする
    Datasets -= Datasets.mean()
    pca = PCA(n_components=3)
    A = Datasets.reshape(n_class * n_sample, pixel_size ** 2)
    B = pca.fit_transform(A)
    U = A.T @ B   # 変換行列U

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

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(epoch):
        ax.cla()
        display_manifold_idx = np.linspace(0, n_class-1, 3, dtype="int64")
        for i in display_manifold_idx:
            transformed_cY = cY[epoch, i] @ U
            ax.scatter(transformed_cY[:, 0], transformed_cY[:, 1], transformed_cY[:, 2])
            # for j in bmu[epoch]:
            #     j = int(j)
            #     ax.text(transformed_cY[j, 0], transformed_cY[j, 1], transformed_cY[j, 2], j)
        # fiberの表示
        for i in np.linspace(0, cCluster_num-1, 5, dtype="int64"):
            transformed_cY = cY[epoch, :, i] @ U
            transformed_cY = transformed_cY[display_manifold_idx]
            ax.plot(transformed_cY[:, 0], transformed_cY[:, 1], transformed_cY[:, 2], color='gray')

        ax.set_xlabel("component1")
        ax.set_ylabel("component2")
        ax.set_zlabel("component3")

        fig.suptitle(
            "epoch {}/{} observation space".format((epoch + 1), nb_epoch),
            fontsize=10,
        )

    ani = anim.FuncAnimation(
        fig, update, interval=interval, frames=nb_epoch, repeat=False
    )
    ani.save("SOM2_PCA.gif", writer='pillow')
    # plt.show()
