import numpy as np
import scipy.spatial.distance as dist
from sklearn.preprocessing import StandardScaler

# U-matrix表示用の値（勾配）を算出
def calc_umatrix(Zeta, X, Z, sigma, clipping_value=[-2.0, 2.0]):
    # H, G, Rの算出
    dist_z = dist.cdist(Zeta, Z, 'sqeuclidean')
    H = np.exp(-dist_z / (2 * sigma * sigma))
    G = H.sum(axis=1)[:, np.newaxis]
    R = H / G

    # V, V_meanの算出
    V = R[:, :, np.newaxis] * (Z[np.newaxis, :, :] - Zeta[:, np.newaxis, :])          # KxNxL
    V_mean = V.sum(axis=1)[:, np.newaxis, :]                                                    # Kx1xL

    # dYdZの算出
    dRdZ = V - R[:, :, np.newaxis] * V_mean                                                     # KxNxL
    dYdZ = np.einsum("knl,nd->kld", dRdZ, X)     # KxLxD
    dYdZ_norm = np.sum(dYdZ ** 2, axis=(1, 2))                                                  # Kx1

    # 表示用の値を算出（標準化）
    sc = StandardScaler()
    dY_std = sc.fit_transform(dYdZ_norm[:, np.newaxis])


    return np.clip(dY_std, clipping_value[0], clipping_value[1])
