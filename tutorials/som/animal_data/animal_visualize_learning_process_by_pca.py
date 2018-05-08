import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../../')

from libs.models.som.som import SOM
from libs.datasets.artificial import animal
from libs.visualization.som.animation_learning_process_3d import anime_learning_process_3d


if __name__ == '__main__':
    nb_epoch = 100
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.4
    tau = 50
    latent_dim = 2
    seed = 1

    title_text="animal_map_learning_process"
    repeat = False
    save_gif = False

    X, labels = animal.load_data()

    X -= X.mean(axis=0)

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution,
              sigma_max=sigma_max, sigma_min=sigma_min, tau=tau)
    som.fit(nb_epoch=nb_epoch)

    anime_learning_process_3d(X=som.X, Y_allepoch=som.history['y'], labels=labels,
                              repeat=repeat, title_text=title_text, save_gif=save_gif)
