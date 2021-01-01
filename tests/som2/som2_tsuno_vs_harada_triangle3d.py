#!/usr/bin/env python
# coding: utf-8

import numpy as np

from libs.models.som2 import SOM2
from tests.som2.som2_harada import SOM2_harada

if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    nb_epoch = 200
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
    theta = np.linspace(-np.pi / 12, np.pi / 12, n_class)
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
