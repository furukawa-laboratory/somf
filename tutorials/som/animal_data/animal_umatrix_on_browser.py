import numpy as np

import sys
sys.path.append('../../')

from libs.models.som import SOM
from libs.visualization.som.Umatrix_on_browser import SOM_Umatrix
from libs.datasets.artificial import animal


if __name__ == '__main__':
    nb_epoch = 50
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.3
    tau = 50
    latent_dim = 2
    seed = 1

    title="animal map"
    umat_resolution = 100 #U-matrix表示の解像度

    X, labels = animal.load_data()

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau)
    som.fit(nb_epoch=nb_epoch)

    Z = som.Z
    sigma = som.history['sigma'][-1]

    som_umatrix = SOM_Umatrix(X=X,
                              Z=Z,
                              sigma=sigma,
                              labels=labels,
                              title_text=title,
                              resolution=umat_resolution)
    som_umatrix.draw_umatrix()