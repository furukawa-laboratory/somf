#!/usr/bin/env python
# coding: utf-8

import numpy as np

from libs.models.som2 import SOM2
from tests.som2.som2_harada import SOM2_harada

if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    nb_epoch = 100
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
    parent_sigma_min = 0.3
    child_sigma_max = 2.0
    child_sigma_min = 0.15
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
    y = [0.5 * x[0], -0.5 * x[1], 0.5 * x[2] ** 2, -0.5 * (x[3] ** 2), x[4] ** 3 - 0.5 * x[4], -x[5] ** 3 + 0.5 * x[5]]
    for n, (a, b) in enumerate(zip(x, y)):
        Datasets[n, :, 0] = a
        Datasets[n, :, 1] = b

    model_tsuno = SOM2(Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZ, cZ)
    model_harada = SOM2_harada(Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZ, cZ)
    model_tsuno.fit(nb_epoch)
    model_harada.fit(nb_epoch)

    print("cZ", np.allclose(model_tsuno.history["cZ"], model_harada.history["cZ"]))
    print("pZ", np.allclose(model_tsuno.history["pZ"], model_harada.history["pZ"]))
    print("cY", np.allclose(model_tsuno.history["cY"], model_harada.history["cY"]))
    print("pY", np.allclose(model_tsuno.history["pY"], model_harada.history["pY"]))
    print("bmu", np.allclose(model_tsuno.history["bmu"], model_harada.history["bmu"]))
    print("bmm", np.allclose(model_tsuno.history["bmm"], model_harada.history["bmm"]))
