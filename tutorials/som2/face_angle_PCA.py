#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import offsetbox
from PIL import Image
from sklearn.decomposition import PCA
from libs.models.som2 import SOM2

if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    nb_epoch = 20
    max_angle = 80
    pixel_size = 64
    n_class = 5
    n_sample = int(2 * max_angle / 5 + 1)
    Dim = pixel_size ** 2
    parent_latent_dim = 2
    child_latent_dim = 1
    parent_resolution = 10
    child_resolution = 50
    pCluster_num = parent_resolution ** parent_latent_dim
    cCluster_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.3
    child_sigma_max = 2.0
    child_sigma_min = 0.15
    parent_tau = nb_epoch * 0.8
    child_tau = nb_epoch * 0.8
    print_n_class = 2    # 表示するクラス数
    print_n_sample = 5   # 表示するサンプル数
    print_n_fiber = 5    # 表示するファイバー数
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
    U = A.T @ B  # 変換行列U

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
    bmu = model.history["bmu"].astype(np.int64)
    bmm = model.history["bmm"].astype(np.int64)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)

    # Create a dummy axes to place annotations to
    ax2 = fig.add_subplot(111, frame_on=False)


    def proj(X, ax1, ax2):
        """ From a 3D point in axes ax1,
            calculate position in 2D in ax2 """
        x, y, z = X
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax1.get_proj())
        return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))


    def image(ax, arr, xy, edgecolor):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=0.3)
        im.image.axes = ax
        ab = offsetbox.AnnotationBbox(im, xy, xybox=(30., 30.),
                                      xycoords='data', boxcoords="offset points",
                                      pad=0.2, arrowprops=dict(arrowstyle="-"), bboxprops=dict(edgecolor=edgecolor))
        ax.add_artist(ab)


    # 色を乱数生成
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def update(epoch):
        ax.cla()
        ax2.cla()

        ax2.axis("off")
        ax2.axis([0, 1, 0, 1])

        # 画像を3次元空間に表示する前に子SOMの描画を行う
        for n, i in enumerate(np.linspace(0, n_class - 1, print_n_class, dtype="int64")):
            trans_cY = cY[epoch, i] @ U
            ax.scatter(trans_cY[:, 0], trans_cY[:, 1], trans_cY[:, 2], c=cycle[n])

        # 3次元空間中に画像を表示
        for n, i in enumerate(np.linspace(0, n_class - 1, print_n_class, dtype="int64")):
            trans_cY = cY[epoch, i] @ U
            for j in np.linspace(0, n_sample - 1, print_n_sample, dtype="int64"):
                s = trans_cY[bmu[epoch, i, j]]
                x, y = proj([s[0], s[1], s[2]], ax, ax2)
                img = Datasets[i, j].reshape(pixel_size, pixel_size)
                image(ax2, img, [x, y], cycle[n])

        # fiberの表示
        idx = np.linspace(0, n_class - 1, print_n_class, dtype="int64")
        for i in np.linspace(0, cCluster_num - 1, print_n_fiber, dtype="int64"):
            trans_cY = cY[epoch, idx, i] @ U
            ax.plot(trans_cY[:, 0], trans_cY[:, 1], trans_cY[:, 2], color='gray')

        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='z', labelsize=7)

        ax.set_xlabel("component1", fontsize=9)
        ax.set_ylabel("component2", fontsize=9)
        ax.set_zlabel("component3", fontsize=9)

        ax.set_title("epoch {}/{} observation space".format((epoch + 1), nb_epoch), fontsize=8)


    def rotate(angle):
        ax.cla()
        ax2.cla()

        ax.view_init(azim=angle)
        ax2.axis("off")
        ax2.axis([0, 1, 0, 1])

        # 画像を3次元空間に表示する前に子SOMの描画を行う
        for n, i in enumerate(np.linspace(0, n_class - 1, print_n_class, dtype="int64")):
            trans_cY = cY[nb_epoch - 1, i] @ U
            ax.scatter(trans_cY[:, 0], trans_cY[:, 1], trans_cY[:, 2], c=cycle[n % 10])

        # 3次元空間中に画像を表示
        for n, i in enumerate(np.linspace(0, n_class - 1, print_n_class, dtype="int64")):
            trans_cY = cY[nb_epoch - 1, i] @ U
            for j in np.linspace(0, n_sample - 1, print_n_sample, dtype="int64"):
                s = trans_cY[bmu[nb_epoch - 1, i, j]]
                x, y = proj([s[0], s[1], s[2]], ax, ax2)
                img = Datasets[i, j].reshape(pixel_size, pixel_size)
                image(ax2, img, [x, y], cycle[n])

        # fiberの表示
        idx = np.linspace(0, n_class - 1, print_n_class, dtype="int64")
        for i in np.linspace(0, cCluster_num - 1, print_n_fiber, dtype="int64"):
            trans_cY = cY[nb_epoch - 1, idx, i] @ U
            ax.plot(trans_cY[:, 0], trans_cY[:, 1], trans_cY[:, 2], color='gray')

        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='z', labelsize=7)

        ax.set_xlabel("component1", fontsize=9)
        ax.set_ylabel("component2", fontsize=9)
        ax.set_zlabel("component3", fontsize=9)

        ax.set_title("observation space", fontsize=8)


    # 学習過程のアニメーション
    ani = anim.FuncAnimation(
        fig, update, interval=interval, frames=nb_epoch, repeat=True
    )
    # 学習後のアニメーション（回転）
    # ani = anim.FuncAnimation(
    #     fig, rotate, interval=interval, frames=np.arange(0, 364, 4), repeat=True
    # )
    ani.save("SOM2_PCA.gif", writer='pillow')
    # plt.show()
