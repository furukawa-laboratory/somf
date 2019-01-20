import numpy as np
import scipy.spatial.distance as dist
from sklearn.preprocessing import StandardScaler

# calculation of  gradient square norm of Kernel Smoothing (Nadaraya-Watson Regression)
# standardization of the values
def calc_grad_norm_of_ks(Zeta, X, Z, sigma, clipping_value=[-2.0, 2.0]):
    # calculate gaussian kernel, etc.
    dist_z = dist.cdist(Zeta, Z, 'sqeuclidean')
    H = np.exp(-dist_z / (2 * sigma * sigma))
    G = H.sum(axis=1)[:, np.newaxis]
    R = H / G

    V = R[:, :, np.newaxis] * (Z[np.newaxis, :, :] - Zeta[:, np.newaxis, :])          # KxNxL
    V_mean = V.sum(axis=1)[:, np.newaxis, :]                                                    # Kx1xL

    # calculate true gradient squared norm
    dRdZ = V - R[:, :, np.newaxis] * V_mean                                                     # KxNxL
    dYdZ = np.einsum("knl,nd->kld", dRdZ, X)     # KxLxD
    dYdZ_norm = np.sum(dYdZ ** 2, axis=(1, 2))                                                  # Kx1

    # standardize mean zero, variance one
    sc = StandardScaler()
    dY_std = sc.fit_transform(dYdZ_norm[:, np.newaxis])

    # clip
    return np.clip(dY_std, clipping_value[0], clipping_value[1])
