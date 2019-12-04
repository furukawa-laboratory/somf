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
        # set parameters of som
        nb_epoch = 50
        resolution = 10
        sigma_max = 2.2
        sigma_min = 0.3
        tau = 50
        latent_dim = 2

        # set parameters of training data
        seed = 1
        n_samples = 500
        n_features = 200

        # generate training data
        random_state = check_random_state(seed=seed)
        # X = random_state.normal(scale=1.0,size=(n_samples,n_features))
        mean = np.zeros(n_features)
        cov = np.diag(np.exp(-np.linspace(0.0,5.0,n_features)**2.0))
        X = random_state.multivariate_normal(mean=mean,cov=cov,size=n_samples)
        X -= np.mean(X,axis=0)

        # initialize som and pickup initial value by PCA
        som = SOM(X, latent_dim=latent_dim, resolution=resolution,
                  sigma_max=sigma_max, sigma_min=sigma_min, tau=tau,
                  init='PCA')
        SOMResult = som.Z

        # calculate init value using different method, np.linalg.svd
        U, S, V = np.linalg.svd(X, full_matrices=False)

        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        V *= signs[:, np.newaxis]

        SVDResult = U[:, :latent_dim] * S[:latent_dim]

        np.testing.assert_allclose(SOMResult, SVDResult/S.max(), rtol=1e-06)

        # calculate init value using different method, np.linalg.eig
        Lambda, V = np.linalg.eig(X.T@X)
        EVDResult = X @ V.real[:, :latent_dim] * np.array([-1.0,1.0])[None,:]

        np.testing.assert_allclose(SOMResult, EVDResult/np.sqrt(Lambda.real.max()), rtol=1e-06)

    def test_transform(self):
        n_distributon = 100
        n_category = 20

        # create categorical distribution
        X_categorical = np.random.rand(n_distributon,n_category)
        X_categorical = X_categorical / X_categorical.sum(axis=1)[:,None]

        np.testing.assert_allclose(X_categorical.sum(axis=1),np.ones(X_categorical.shape[0]))

        # fit
        som_categorical = SOM(X_categorical,latent_dim=2,resolution=50,sigma_max=2.0,sigma_min=0.3,tau=50,metric="KLdivergence")
        som_categorical.fit(50)
        Z_fit = som_categorical.Z
        Z_transformed = som_categorical.transform(X_categorical)

        np.testing.assert_allclose(Z_transformed,Z_fit)

        # confirm to multi variable dataset
        n_samples = 100
        n_features = 20

        X_multi_variate = np.random.normal(0.0,1.0,(n_samples,n_features))

        # fit
        som_multi_variate = SOM(X_multi_variate,latent_dim=2,resolution=50,sigma_max=2.0,sigma_min=0.2,tau=50,metric="sqeuclidean")
        som_multi_variate.fit(10)
        Z_fit = som_multi_variate.Z
        Z_transformed = som_multi_variate.transform(X_multi_variate)

        np.testing.assert_allclose(Z_fit,Z_transformed)




if __name__ == "__main__":
    unittest.main()
