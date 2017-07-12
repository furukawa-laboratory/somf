import numpy as np
import scipy.spatial.distance as dist


class KSE(object):
    def __init__(self, X, latent_dim, init='random', choice_estimate='MAP',choice_beta='type1'):
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

        self.choice_estimate = choice_estimate
        self.choice_beta = choice_beta

        if self.choice_estimate == 'MAP':
            if self.choice_beta not in ['type1','type2']:
                raise ValueError("invalid combination of choice_estimate and choice_beta: {0},{1}".format(self.choice_estimate,self.choice_beta))
        elif self.choice_estimate == 'distribution':
            if self.choice_beta not in ['type3']:
                raise ValueError("invalid combination of choice_estimate and choice_beta: {0},{1}".format(self.choice_estimate,self.choice_beta))
        else:
            raise ValueError("invalid choice_estimate: {}".format(choice_estimate))

        self.history = {}

    def fit(self, nb_epoch=100, epsilon=0.5, gamma=1.0, sigma=30.0):

        K = self.X @ self.X.T
        X2 = np.diag(K)[:, None]
        alpha = 1.0 / (sigma ** 2)

        self.history['z'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y'] = np.zeros((nb_epoch, self.N, self.D))

        for epoch in range(nb_epoch):
            Delta = self.Z[:, None, :] - self.Z[None, :, :]
            Dist = np.sum(np.square(Delta), axis=2)
            if self.choice_estimate is 'MAP':
                H = np.exp(-0.5 * gamma * Dist)
            elif self.choice_estimate is 'distribution':
                H = (1.0 / np.sqrt(2) ) * np.exp(-0.25 * gamma * Dist)
            H -= H * np.identity(self.N)
            Hprime = H

            G = np.sum(H, axis=1)[:, None]
            GInv = 1 / G
            R = H * GInv
            Rprime = Hprime * GInv

            Y = R @ self.X

            U = R - np.identity(self.N)
            Phi = U @ K
            PhiBar = np.sum(R * Phi, axis=1)[:, None]
            E = np.diag(U @ K @ U.T)[:, None]

            if self.choice_beta == 'type1':
                Y2 = np.sum(np.square(Y), axis=1)[:, None]
                beta0 = np.sum(G) / np.sum(G * (X2 - Y2))
            elif self.choice_beta == 'type2':
                beta0 = (self.N * self.D) / np.sum(E)
            elif self.choice_beta == 'type3':
                Delta = self.Z[:, None, :] - self.Z[None, :, :]
                Dist = gamma/(2 * np.pi) * np.sum(np.square(Delta), axis=2)
                Q = ( ( gamma / (2.0*np.pi) ) ** ( self.L / 2.0 ) ) * np.exp(-0.5 * gamma * Dist)
                DistXY = dist.cdist(self.X,self.Y,'sqeuclidean')
                beta0 = ( self.N * self.D ) / np.sum( Q * DistXY )
            beta = (G / (1 + G)) * beta0

            A = Rprime * (beta * (Phi - PhiBar))
            A += Rprime * (0.5 * (beta * E - 1.0) / (1.0 + G))

            dZ = np.sum((A + A.T)[:, :, None] * Delta, axis=1)
            dZ -= (alpha / gamma) * self.Z

            self.Z += epsilon * dZ

            self.history['z'][epoch] = self.Z
            self.history['y'][epoch] = Y

        return self.history
