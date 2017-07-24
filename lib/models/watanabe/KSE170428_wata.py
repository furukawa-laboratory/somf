import numpy as np
import scipy.spatial.distance as dist
# import sys
import math
from tqdm import tqdm
from sklearn import linear_model

#5/24との比較検証用にhistoryを追加
class KSE170428_wata_gamma:
    def __init__(self, X, L, M, Z0=None):
        self.X = X
        [self.N, self.D] = X.shape
        self.K = X @ X.T
        self.L = L
        self.M = M
        if Z0 is None:
            self.Z = np.random.rand((self.N, self.L))
        self.Z = Z0.copy()
        self.history = {}
        self.x_sqnorms = np.diag(self.K)[:, np.newaxis]
        self.gamma = 1.0
        self.DistXX = dist.cdist(self.X, self.X, 'sqeuclidean')#NxK行列
        self.alpha = 1.0

    def fit(self, Epoch=100, epsilon=0.5, isLOOCV=True):
        self.history["z"] = np.zeros((Epoch, self.N, self.L))
        self.history["f"] = np.zeros((Epoch, self.M, self.D))
        self.history["beta0"] = np.zeros(Epoch)
        self.history["gamma"] = np.zeros(Epoch)
        I = np.eye(self.N)
        for e in tqdm(range(Epoch)):
            # E-step
            DistZZ = dist.cdist(self.Z, self.Z, 'sqeuclidean')
            H = np.exp(-0.5 * self.gamma * DistZZ)
            if isLOOCV:
                H = H - I  # LOOCV
            g = H.sum(axis=1)[:, np.newaxis]
            g_inv = 1/g
            R = H * g_inv  # KxN行列
            Phi = (R - I) @ self.K  # KxN行列
            Phibar = (R * Phi).sum(axis=1)[:, np.newaxis]  # Kx1
            delta = self.Z[:, np.newaxis, :] - self.Z[np.newaxis, :, :]
            # beta0,betaの計算
            Y = R @ self.X
            beta0 = H.sum() / (H * dist.cdist(Y, self.X, 'sqeuclidean')).sum()
            beta = beta0 * g / (1 + g)
            self.history['beta0'][e] = beta0

            # E = np.sum((Y - self.X)**2,axis=1)[:,np.newaxis]#Kx1
            E = np.diag((R - I) @ self.K @ (R - I).T)[:, np.newaxis]  # Kx1
            dB = 0.5 * (beta * E - 1) / (1 + g)
            dE = beta * (Phi - Phibar)
            A = R * (dE + dB)
            # A = beta * R * (Phi - Phibar) + 0.5 * R * (beta * E - 1) / (1 + gtilda)
            dZ = ((A + A.T)[:, :, np.newaxis] * delta).sum(axis=1) - (self.alpha/self.gamma) * self.Z

            self.Z = self.Z + epsilon * dZ
            self.mat = dZ
            self.history['z'][e] = self.Z


            #if e%10==0 and e>300:
            if e%50==49:
                x = DistZZ.reshape(self.N*self.N,1)
                y = beta0*self.DistXX.reshape(self.N * self.N,1)
                h = H.reshape(self.N*self.N)
                LR = linear_model.LinearRegression()
                LR.fit(x,y,sample_weight=h)
                #LR.fit(x,y)
                self.gamma = np.float(LR.coef_)
            self.history['gamma'][e] = self.gamma


            # M-step
            if self.L == 1:
                Zeta = np.linspace(self.Z.min(), self.Z.max(), self.M)
                Zeta = Zeta[:, np.newaxis]
                Dist = dist.cdist(Zeta, self.Z, 'sqeuclidean')
                H = np.exp(-0.5 * self.gamma *Dist)
                G = H.sum(axis=1)[:, np.newaxis]
                R = H / G
                self.history['f'][e] = R @ self.X
            elif self.L == 2:
                Zeta = np.meshgrid(np.linspace(self.Z[:, 0].min(), self.Z[:, 0].max(), np.sqrt(self.M)),
                                   np.linspace(self.Z[:, 1].min(), self.Z[:, 1].max(), np.sqrt(self.M)))
                Zeta = np.dstack(Zeta).reshape(self.M, self.L)
                Dist = dist.cdist(Zeta, self.Z, 'sqeuclidean')
                H = np.exp(-0.5 * self.gamma*Dist)
                G = H.sum(axis=1)[:, np.newaxis]
                R = H / G
                self.history['f'][e] = R @ self.X
                # print(e)
