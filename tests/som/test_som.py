import unittest

import numpy as np

from libs.models.som import SOM
from libs.models.som_use_for import SOMUseFor

from sklearn.utils import check_random_state

class TestSOM(unittest.TestCase):
    def test_numpy_vs_usefor(self):
        N = 100
        D = 3
        L = 2
        resolution = 10
        M = resolution ** L
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Zinit = np.random.rand(N,L)

        nb_epoch = 200
        SIGMA_MAX = 2.2
        SIGMA_MIN = 0.1
        TAU = 50
        som_numpy = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)
        som_use_for = SOMUseFor(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=Zinit)

        som_numpy.fit(nb_epoch=nb_epoch)
        som_use_for.fit(nb_epoch=nb_epoch)

        np.testing.assert_allclose(som_numpy.history['y'],som_use_for.history['y'])

    def test_init(self):
        N = 100
        D = 3
        L = 2
        resolution = 10
        seed = 100
        np.random.seed(seed)
        X = np.random.normal(0, 1, (N, D))
        Zinit = np.random.rand(N,L)

        inits = ['random', 'random_bmu', Zinit]

        SIGMA_MAX = 2.2
        SIGMA_MIN = 0.1
        TAU = 50

        for init in inits:
            som = SOM(X,L,resolution,SIGMA_MAX,SIGMA_MIN,TAU,init=init)

    def test_init_pca(self):
        nb_epoch = 50
        resolution = 10
        sigma_max = 2.2
        sigma_min = 0.3
        tau = 50
        latent_dim = 2
        seed = 1

        X = [[1,2,3],[2,2,2],[5,1,3]]
        X -= np.mean(X,axis=0)

        n_components = latent_dim

        np.random.seed(seed)

        som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau,
                  init='PCA')
        som.fit(nb_epoch=nb_epoch)

        n_samples, n_features = X.shape

        PCAResult, zeta = som.history['z0_zeta0']

        U, S, V = np.linalg.svd(X, full_matrices=False)

        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        V *= signs[:, np.newaxis]

        U = U[:, :n_components]

        # U *= np.sqrt(X.shape[0] - 1)
        U *= S[:n_components]

        SVDResult = U

        np.testing.assert_allclose(PCAResult, SVDResult, rtol=1e-06)



if __name__ == "__main__":
    unittest.main()
