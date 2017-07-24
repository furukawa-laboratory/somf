import numpy as np
import math
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
from tqdm import tqdm

class KSE_fit_error2(object):
    def __init__(self, X, L,M, Z0=None):
        self.X = X
        [self.N,self.D] = X.shape
        self.K = X @ X.T

        self.L = L
        self.M = M
        if Z0 is None:
            self.Z = np.random.rand((self.N,self.L))
        self.Z = Z0.copy()
        self.Gamma = 1
        self.history={}

    def fit(self, Epoch=100, epsilon=0.5,blank=1,calcFunction=False,scale=1):
        self.history["z"] = np.zeros((Epoch, self.N, self.L))
        if calcFunction is True:
            self.history["f"] = np.zeros((Epoch, self.M, self.D))
        self.history["y"] = np.zeros((Epoch, self.N, self.D))
        self.history["Rho"] = np.zeros((Epoch, self.N))
        self.history["Rad"] = np.zeros(Epoch)
        self.history["beta"] = np.zeros(Epoch)
        self.history["Gamma"] = np.zeros(Epoch)
        I = np.identity(self.N)
        Rho = np.ones((1,self.N))
        self.Alpha = scale
        self.Gamma = scale
        self.Z = self.Z*1/np.sqrt(scale)
        ErrorO = dist.cdist(self.X, self.X, 'sqeuclidean')

        for e in tqdm(range(Epoch)):
            # 写像の更新
            if calcFunction is True:
                if self.L == 1:
                    Zeta = np.linspace(self.Z.min() - blank, self.Z.max() + blank, self.M)
                    Zeta = Zeta[:, np.newaxis]
                    Dist = dist.cdist(Zeta, self.Z, 'sqeuclidean')
                    H = np.exp(-0.5 * self.Gamma * Dist)
                    G = H.sum(axis=1)[:, np.newaxis]
                    R = H / G
                    self.history['f'][e] = R @ self.X
                elif self.L == 2:
                    Zeta = np.meshgrid(np.linspace(self.Z[:, 0].min() - blank, self.Z[:, 0].max() + blank, math.sqrt(self.M)),
                                       np.linspace(self.Z[:, 1].min() - blank, self.Z[:, 1].max() + blank, math.sqrt(self.M)))
                    Zeta = np.dstack(Zeta).reshape(self.M, self.L)
                    Dist = dist.cdist(Zeta, self.Z, 'sqeuclidean')
                    H = np.exp(-0.5 * self.Gamma * Dist)
                    G = H.sum(axis=1)[:, np.newaxis]
                    R = H / G
                    self.history['f'][e] = R @ self.X

            # Rの更新
            Dist = dist.cdist(self.Z, self.Z, 'sqeuclidean')
            H = np.exp(-0.5 * self.Gamma * Dist)
            H = H - I
            Hprime = H
            G = H.sum(axis=1)[:,np.newaxis]
            Ginv = 1/G
            R = H * Ginv
            Rprime = Ginv * Hprime
            Y = R @ self.X
            self.history["y"][e] = Y

            # betaの更新
            Q = G/G.sum()
            beta0 = 1.0/(Q.T @ ( np.diag(self.K) - np.diag(R @ self.K @ R.T) ))
            beta = G/(1+G)*beta0
            self.history["beta"][e] = 1/np.sqrt(beta0)

            # Zの更新
            Phi = (R - I) @ self.K
            Phibar = (R * Phi).sum(axis=1)[:,np.newaxis]
            delta = self.Z[:,np.newaxis,:]-self.Z[np.newaxis,:,:]
            E = np.diag((R-I) @ self.K @ (R-I).T)[:,np.newaxis]
            dEdZ = Rprime * ( beta * (Phi - Phibar) + 0.5 * (beta * E - 1) / (1 + G))
            dZ = ((dEdZ + dEdZ.T)[:, :, np.newaxis] * delta).sum(axis=1) - self.Alpha/self.Gamma * self.Z
            self.Z += epsilon * dZ

            # Gammaの更新
            if e%10==9:
                ErrorL = Dist
                ErrorL2 = ErrorL.reshape(self.N*self.N)[:,np.newaxis]
                one = np.ones(self.N*self.N)[:,np.newaxis]
                input = np.concatenate((ErrorL2, one), axis=1)
                output = (beta0*ErrorO).reshape(self.N*self.N)
                Hvec = H.reshape(self.N*self.N)[:,np.newaxis]
                w = np.linalg.inv((input[:,:,np.newaxis] * Hvec[:,:,np.newaxis] * input[:,np.newaxis,:]).sum(axis=0)) @ (input * Hvec * output[:,np.newaxis]).sum(axis=0)
                self.Gamma = w[0]
            # if e== Epoch/2:
            #     self.Gamma += 100

            self.history['z'][e] = self.Z
            self.history["Rad"][e] = 1/np.sqrt(self.Gamma)
            self.history["Rho"][e]=Rho[0,:]
            self.history["Gamma"][e] = self.Gamma

