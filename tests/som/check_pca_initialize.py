import numpy as np
import sys

sys.path.append('../../')

from libs.models.som import SOM
from libs.datasets.artificial import animal


if __name__ == '__main__':
    nb_epoch = 50
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.3
    tau = 50
    latent_dim = 2
    seed = 1

    title = "animal map"
    umat_resolution = 100  # U-matrix表示の解像度

    X, labels = animal.load_data()
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

    # 一致しなかったコード
    # S = np.zeros((n_samples, n_features))
    # S_diag = np.diag((S))
    # S[:n_samples, :n_samples] = np.diag(S)

    # left = U.T @ X
    # right = S @ V.T
    #
    # left = U @ S
    # right = X @ V

    # np.testing.assert_allclose(right,left,rtol=1e-06)
    # np.testing.assert_allclose(Z0,left,rtol=1e-06)
    # np.testing.assert_allclose(right,Z0,rtol=1e-06)

    # S = np.zeros((n,p))
    # S_diag = np.diag((s))
    # S[:n,:n] = np.diag(s)
    # S_T = S.T
    # S_for_2 = S_T[:, :2].T
    # # SVD_Reduction = X @ S_for_2
    #
    # V_for_2 = V[:, :2]
    # SVD_Reduction = S_for_2 @ V

