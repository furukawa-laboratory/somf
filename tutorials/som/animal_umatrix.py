import numpy as np
from som import SOM
from tqdm import tqdm
from Umatrix import SOM_Umatrix

if __name__ == '__main__':
    T = 300
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.4
    tau = 50
    latent_dim = 2
    seed = 1

    title="animal map"
    umat_resolution = 100 #U-matrix表示の解像度

    #X = np.loadtxt("../data/kura.txt")  # reading observation data
    X = np.loadtxt('../../data/animal/features.txt')
    labels = np.genfromtxt('../../data/animal/labels.txt',dtype=str)
    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau)
    for t in tqdm(range(T)):
        som.learning(t)

    som_umatrix = SOM_Umatrix(z=som.Z, x=X, resolution=umat_resolution, sigma=sigma_min, labels=labels)
    som_umatrix.draw_umatrix()