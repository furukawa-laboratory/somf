import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../../')

from libs.models.som.som import SOM
from libs.visualization.som.Umatrix import SOM_Umatrix
from libs.datasets.artificial import coffee


if __name__ == '__main__':
    nb_epoch = 300
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.4
    tau = 50
    latent_dim = 2
    seed = 1

    title="coffee map"
    umat_resolution = 100 #U-matrix表示の解像度

    X, labels = coffee.load_data()

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau)
    som.fit(nb_epoch=nb_epoch)

    som_umatrix = SOM_Umatrix(z=som.Z, x=X, resolution=umat_resolution, sigma=sigma_min, labels=labels)
    som_umatrix.draw_umatrix()