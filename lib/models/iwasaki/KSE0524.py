import numpy as np
import scipy.spatial.distance as dist
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

class KSE(object):
    def __init__(self, X, latent_dim, init='random'):
        self.name = 'KSE0524'
        self.f_options = ['point', 'distribution']
        self.z_options = ['point', 'distribution']
        self.divdz = True
        self.divgamma = False
        if self.divgamma:
            self.name += "_divD"

        self.X = X.copy()
        self.nb_samples = X.shape[0]
        self.input_dim = X.shape[1]
        self.latent_dim = latent_dim

        self.Z = None
        if isinstance(init, str) and init in 'random':
            self.Z = np.random.normal(0, 0.1, (self.nb_samples, self.latent_dim))
        elif isinstance(init, np.ndarray) and init.shape == (self.nb_samples, self.latent_dim):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        self.Alpha = 1.0
        self.Gamma = 1.0

        if latent_dim == 1:
            self.topology_name = "LINE"
        elif latent_dim == 2:
            self.topology_name = "PLANE"
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))

        self.history = {}

    def fit(self, nb_epoch: int = 100, z_epsilon=0.3, gamma_epsilon=0.3,
            loocv=True, f_resolution=None,
            z_option='point', f_option='point'):
        """
        
        :param nb_epoch: 
        :param z_epsilon: 
        :param gamma_epsilon: 
        :param loocv: 
        :param f_resolution: 
        :param z_option: 
        :param f_option: 
        :return: 
        """

        if z_option not in self.z_options:
            ValueError("{0} not in {1}".format(z_option, self.z_options))
        if f_option not in self.f_options:
            ValueError("{0} not in {1}".format(f_option, self.f_options))

        self.history["z"] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['y'] = np.zeros((nb_epoch, self.nb_samples, self.input_dim))
        self.history['beta'] = np.zeros(nb_epoch)
        self.history['beta_n'] = np.zeros((nb_epoch, self.nb_samples))
        self.history['gamma'] = np.zeros(nb_epoch)
        self.history['zlength'] = np.zeros(nb_epoch)

        K = np.dot(self.X, self.X.T)
        X2 = np.diag(K)
        Xn = np.sum(np.square(self.X[:, np.newaxis, :] - self.X[np.newaxis, :, :]), axis=2)
        DistX = Xn.reshape(self.nb_samples**2, 1)

        slr = LinearRegression()
        for epoch in tqdm(range(nb_epoch)):
            Delta = self.Z[:, np.newaxis, :] - self.Z[np.newaxis, :, :]
            DistZ = np.sum(np.square(Delta), axis=2)

            H = np.exp(-0.5 * self.Gamma * DistZ)
            H -= H * np.identity(self.nb_samples)
            Hp = H

            if z_option in ['point']:
                Q = H
            else:
                Q = H ** 2
            Q -= Q * np.identity(self.nb_samples)
            Qsum = Q.sum()

            G = np.sum(H, axis=1)[:, np.newaxis]
            GInv = np.reciprocal(G)
            R = H * GInv
            Rp = Hp * GInv

            # calc Beta
            Y = np.dot(R, self.X)

            # Eij = np.sum(np.square(Y[:, None, :] - self.X[None, :, :]), axis=2)
            # beta0 = Qsum * self.input_dim / np.sum(Q * Eij)

            beta0 = Qsum * self.input_dim / np.sum(Q * Xn)

            if z_option in ['point']:
                beta = G / (1 + G) * beta0
            else:
                beta = G / (2.0 ** (self.latent_dim / 2.0) + G) * beta0

            U = R - np.identity(self.nb_samples)
            Phi = np.dot(U, K)
            PhiBar = np.sum(R * Phi, axis=1)[:, np.newaxis]
            # E = np.sum((Y - self.X) ** 2, axis=1)[:, np.newaxis]
            # for pair check
            E = np.diag(U @ K @ U.T)[:, np.newaxis]

            A = Rp * (beta * (Phi - PhiBar) + 0.5 * (beta * E - self.input_dim) / (1.0 + G))

            if self.divdz:
                dZ = np.sum((A + A.T)[:, :, np.newaxis] * Delta, axis=1)
                dZ -= (self.Alpha / self.Gamma) * self.Z
            else:
                dZ = np.sum((A + A.T)[:, :, np.newaxis] * Delta * self.Gamma, axis=1)
                dZ -= self.Alpha * self.Z

            self.Z += (z_epsilon / self.input_dim) * dZ

            # calc gamma
            if epoch > 300 and epoch % 50 == 0:
                _DistZ = DistZ.flatten()[:, None]
                Weight = Q.flatten()
                slr.fit(_DistZ, beta0 * DistX, Weight)
                if self.divgamma:
                    self.Gamma = slr.coef_ / self.input_dim
                else:
                    self.Gamma = slr.coef_
                # print(self.Gamma)

            # if epoch == 500:
            #     self.Gamma = 1000

            # save
            self.history['z'][epoch] = self.Z
            self.history['zlength'][epoch] = np.max(self.Z) - np.min(self.Z)
            self.history['y'][epoch] = Y
            self.history['beta'][epoch] = beta0
            self.history['beta_n'][epoch] = beta.flatten()
            self.history['gamma'][epoch] = self.Gamma

        # save f
        if f_resolution is not None:
            Zeta = create_zeta(self.topology_name, f_resolution)
            nb_nodes = Zeta.shape[0]
            self.history['f'] = np.zeros((nb_epoch, nb_nodes, self.input_dim))
            for epoch in range(nb_epoch):
                Z = self.history['z'][epoch]
                Gamma = self.history['gamma'][epoch]
                Zeta = create_zeta(self.topology_name, f_resolution, Z.min(), Z.max())
                Dist = dist.cdist(Zeta, Z, 'sqeuclidean')

                H = np.exp(-0.5 * Gamma * Dist)
                # H = (2.0 ** (-self.latent_dim/2.0)) * np.exp(-0.25 * Gamma * Dist)

                G = H.sum(axis=1)[:, np.newaxis]
                R = H / G
                self.history['f'][epoch] = R @ self.X

        return self.history


def create_zeta(zeta_min, zeta_max, latent_dim, resolution):
    mesh1d, step = np.linspace(zeta_min, zeta_max, resolution, endpoint=False, retstep=True)
    mesh1d += step / 2.0
    if latent_dim == 1:
        Zeta = mesh1d
    elif latent_dim == 2:
        Zeta = np.meshgrid(mesh1d, mesh1d)
    else:
        raise ValueError("invalid latent dim {}".format(latent_dim))
    Zeta = np.dstack(Zeta).reshape(-1, latent_dim)
    return Zeta