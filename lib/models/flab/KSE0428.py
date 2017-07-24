import numpy as np
import scipy.spatial.distance as dist
from sklearn import linear_model

class KSE(object):
    def __init__(self, X, latent_dim, init='random'):
        self.X = X.copy()
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.L = latent_dim

        self.Z = None
        if isinstance(init, str) and init in 'random':
            self.Z = np.random.normal(0, 0.1, (self.N, self.L))
        elif isinstance(init, np.ndarray) and init.shape == (self.N, self.L):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        self.history = {}

    def fit(self, nb_epoch=100, epsilon=0.5, gamma=1.0, sigma=30.0, time_gamma_update=50):

        K = self.X @ self.X.T
        X2 = np.diag(K)[:, None]
        alpha = 1.0 / (sigma ** 2)
        DistX = dist.cdist(self.X, self.X, 'sqeuclidean')

        self.nb_epoch = nb_epoch

        self.history['z'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y'] = np.zeros((nb_epoch, self.N, self.D))
        self.history['gamma'] = np.zeros(nb_epoch)
        self.history['beta'] = np.zeros(nb_epoch)

        for epoch in range(nb_epoch):
            Delta = self.Z[:, None, :] - self.Z[None, :, :]
            Dist = np.sum(np.square(Delta), axis=2)
            H = np.exp(-0.5 * gamma * Dist)
            H -= H * np.identity(self.N)
            Hprime = H
            G = np.sum(H, axis=1)[:, None]
            GInv = 1 / G
            R = H * GInv
            Rprime = Hprime * GInv

            Y = R @ self.X
            Y2 = np.sum(np.square(Y), axis=1)[:, None]
            beta0 = np.sum(G) / np.sum(G * (X2 - Y2))
            beta = (G / (1 + G)) * beta0

            U = R - np.identity(self.N)
            Phi = U @ K
            PhiBar = np.sum(R * Phi, axis=1)[:, None]
            E = np.diag(U @ K @ U.T)[:, None]

            A = Rprime * (beta * (Phi - PhiBar))
            A += Rprime * (0.5 * (beta * E - 1.0) / (1.0 + G))

            dZ = np.sum((A + A.T)[:, :, None] * Delta, axis=1)
            dZ -= (alpha / gamma) * self.Z

            self.Z += epsilon * dZ

            if e!=0 and e % time_gamma_update == 0:
                x = Dist.reshape(self.N*self.N,1)
                y = beta0*DistX.reshape(self.N * self.N,1)
                h = H.reshape(self.N*self.N)
                LR = linear_model.LinearRegression()
                LR.fit(x,y,sample_weight=h)
                #LR.fit(x,y)
                gamma = np.float(LR.coef_)

            self.history['z'][epoch] = self.Z
            self.history['y'][epoch] = Y
            self.history['gamma'][epoch] = gamma
            self.history['beta'][epoch] = beta0

        return self.history

    def calcF(self, resolution, size='auto'):
        """
        :param resolution:
        :param size:
        :return:
        """
        self.resolution = resolution
        Zeta = create_zeta(-1, 1, self.L, resolution)
        M = Zeta.shape[0]

        self.history['f'] = np.zeros((self.nb_epoch, M, self.D))

        for epoch in range(self.nb_epoch):
            Z = self.history['z'][epoch]
            gamma = self.history['gamma'][epoch]
            if size == 'auto':
                Zeta = create_zeta(Z.min(), Z.max(), self.L, resolution)
            else:
                Zeta = create_zeta(size.min(), size.max(), self.L, resolution)

            Dist = dist.cdist(Zeta, Z, 'sqeuclidean')

            H = np.exp(-0.5 * gamma * Dist)
            G = np.sum(H, axis=1)[:, None]
            GInv = np.reciprocal(G)
            R = H * GInv

            Y = np.dot(R, self.X)

            self.history['f'][epoch] = Y


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
