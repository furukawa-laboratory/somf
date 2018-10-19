import numpy as np
import sys
sys.path.append('../../')
from libs.models.som import SOM
from libs.visualization.som.animation_learning_process_3d import anime_learning_process_3d
from libs.datasets.artificial.kura import create_data
from libs.visualization.som.Umatrix import SOM_Umatrix
if __name__ == '__main__':
    nb_epoch = 150
    resolution = 20
    SIGMA_MAX = 2.2
    SIGMA_MIN = 0.1
    TAU = 50
    latent_dim = 2

    x_sigma = 0.1

    nb_samples = 1000
    seed = 1

    # X = np.loadtxt("../../data/kura.txt")  # reading observation data
    np.random.seed(seed)
    X = create_data(nb_samples)
    #X += np.random.normal(0,0.1,X.shape)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, tau=TAU)

    som.fit(nb_epoch=nb_epoch)

    # anime_learning_process_3d(X=som.X, Y_allepoch=som.history['y'])
    umat_resolution = resolution
    title = 'kura'
    som_umatrix = SOM_Umatrix(X=som.X,
                              Z=som.Z,
                              sigma=SIGMA_MIN,
                              title_text=title,
                              resolution=umat_resolution)
    som_umatrix.draw_umatrix()
