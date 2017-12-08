import numpy as np
import sys
sys.path.append('../../')
from libs.models.som.som import SOM
from libs.visualization.som.animation_reference_vector3d import anime_reference_vector_3d
from libs.datasets.artificial.kura import create_data
if __name__ == '__main__':
    nb_epoch = 300
    resolution = 20
    SIGMA_MAX = 2.2
    SIGMA_MIN = 0.1
    TAU = 50
    latent_dim = 2

    nb_samples = 300
    seed = 1

    # X = np.loadtxt("../../data/kura.txt")  # reading observation data
    np.random.seed(seed)
    X = create_data(nb_samples)
    D = X.shape[1]

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, tau=TAU)

    som.fit(nb_epoch=nb_epoch)

    anime_reference_vector_3d(X=som.X, allY=som.history['y'])
