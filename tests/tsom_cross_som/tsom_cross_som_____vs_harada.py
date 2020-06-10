#!/usr/bin/env python
# coding: utf-8

import numpy as np

from libs.models.tsom_cross_som import TSOM_cross_SOM
from libs.datasets.artificial.kura_tsom import load_kura_tsom

if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    nb_epoch = 100
    n_class = 6
    n_sample_1 = 10
    n_sample_2 = 10
    Dim = 3
    parent_latent_dim = 2
    child_latent_dim = [1, 1]
    parent_resolution = 8
    child_resolution = [100, 100]
    # pCluster_num = parent_resolution ** parent_latent_dim
    # cCluster_num = child_resolution ** child_latent_dim
    parent_sigma_max = 2.0
    parent_sigma_min = 0.3
    child_sigma_max = 2.0
    child_sigma_min = 0.15
    parent_tau = nb_epoch
    child_tau = nb_epoch
    interval = 500

    # データ生成
    Datasets = np.zeros((n_class, n_sample_1, n_sample_2, Dim))
    # x = np.random.rand(n_class, n_sample) * 2 - 1
    # x = np.sort(x)
    # y = [0.5 * x[0], -0.5 * x[1], 0.5 * x[2] ** 2, -0.5 * (x[3] ** 2), x[4] ** 3 - 0.5 * x[4], -x[5] ** 3 + 0.5 * x[5]]
    # for n, (a, b) in enumerate(zip(x, y)):
    #     Datasets[n, :, 0] = a
    #     Datasets[n, :, 1] = b
    X = load_kura_tsom(xsamples=10,ysamples=10)
    for n in range(n_class):
        Datasets[n, :, :, :] = X

    # ZetaとZの初期値を生成
    # if parent_latent_dim == 2:
    #     pZ = np.random.normal(size=(n_class, parent_latent_dim), loc=0, scale=0.01)
    # else:
    #     pZ = np.random.normal(size=(n_class, 1), loc=0.0, scale=0.01)
    #
    # if child_latent_dim == 2:
    #     cZ = np.random.normal(size=(n_sample_1, n_sample_2, child_latent_dim), loc=0, scale=0.01)
    # else:
    #     cZ = np.random.normal(size=(n_sample_1, n_sample_2, 1), loc=0.0, scale=0.01)
    pZ = None
    cZ = None


    # model_tsuno = SOM2(Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
    #              parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
    #              parent_tau, child_tau, pZ, cZ)
    model_harada = TSOM_cross_SOM(Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZ, cZ)
    # model_tsuno.fit(nb_epoch)
    model_harada.fit(nb_epoch)

    # print("cZ", np.allclose(model_tsuno.history["cZ"], model_harada.history["cZ"]))
    # print("pZ", np.allclose(model_tsuno.history["pZ"], model_harada.history["pZ"]))
    # print("cY", np.allclose(model_tsuno.history["cY"], model_harada.history["cY"]))
    # print("pY", np.allclose(model_tsuno.history["pY"], model_harada.history["pY"]))
    # print("bmu", np.allclose(model_tsuno.history["bmu"], model_harada.history["bmu"]))
    # print("bmm", np.allclose(model_tsuno.history["bmm"], model_harada.history["bmm"]))
