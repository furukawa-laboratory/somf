import numpy as np
import sys
sys.path.append('../../')
from libs.models.som import SOM
from libs.visualization.som.animation_learning_process_3d import anime_learning_process_3d
from libs.datasets.artificial.kura import create_data
if __name__ == '__main__':
    nb_epoch = 50
    resolution = 20
    SIGMA_MAX = 2.2
    SIGMA_MIN = 0.2
    TAU = 50
    latent_dim = 2

    noise_std = 0.1 # standard deviation of noise added to data

    nb_samples = 500
    seed = 1

    title_text = 'som_fit_kura'
    repeat = False
    save_gif = False

    np.random.seed(seed)
    X = create_data(nb_samples)
    X += np.random.normal(0,noise_std,X.shape) # add gaussian noise

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, tau=TAU)

    som.fit(nb_epoch=nb_epoch)

    anime_learning_process_3d(X=som.X, Y_allepoch=som.history['y'],
                              title_text=title_text,repeat=repeat,save_gif=save_gif)
