import numpy as np

import sys
sys.path.append('../../')

from libs.models.som import SOM
from libs.visualization.som.Grad_norm import SOM_Umatrix
from libs.datasets.artificial import coffee


if __name__ == '__main__':
    nb_epoch = 50
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.3
    tau = 50
    latent_dim = 2
    seed = 10

    title="coffee U-matrix"
    umat_resolution = 10 #U-matrix表示の解像度
    interpolation_method = 'spline36'

    X, labels = coffee.load_data()

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau)
    som.fit(nb_epoch=nb_epoch)

    som_umatrix = SOM_Umatrix(Z=som.history['z'], X=X, resolution=umat_resolution,
                              sigma=som.history['sigma'], labels=labels,
                              title_text=title,
                              interpolation_method=interpolation_method)
    som_umatrix.draw_umatrix()