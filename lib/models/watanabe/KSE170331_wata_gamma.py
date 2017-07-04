import numpy as np
import scipy.spatial.distance as dist
import math
from tqdm import tqdm

class KSE170331_wata_gamma:
    def __init__(self, X, L,M, Z0=None):
        self.X = X
        [self.N,self.D] = X.shape
        self.K = X @ X.T
        self.L = L
        self.M = M
        if Z0 is None:
            self.Z = np.random.rand((self.N,self.L))
        self.Z = Z0.copy()
        self.history={}
        self.x_sqnorms = np.diag(self.K)[:, np.newaxis]
        self.gamma = 1.0

    def fit(self, Epoch=100, epsilon=0.5, alpha = 1/(30**2)):
        self.alpha = alpha
        self.history["z"] = np.zeros((Epoch, self.N, self.L))
        self.history["f"] = np.zeros((Epoch, self.M, self.D))
        self.history["zeta"] = np.zeros((Epoch, self.M, self.L))
        self.history["beta0"] = np.zeros(Epoch)
        self.history["g"] = np.zeros((Epoch,self.M))
        self.history["fdelta"] = np.zeros((Epoch,self.M))
        I = np.eye(self.N)
        for e in tqdm(range(Epoch)):
            #E-step
            DistZZ = dist.cdist(self.Z, self.Z, 'sqeuclidean')
            H = np.exp(-0.5*self.gamma*DistZZ)
            H = H - I#LOOCV
            g = H.sum(axis=1)[:,np.newaxis]
            R = H/g#KxN行列
            Phi = (R - I) @ self.K#KxN行列
            Phibar = (R*Phi).sum(axis=1)[:,np.newaxis]#Kx1
            delta = self.Z[:, np.newaxis, :] - self.Z[np.newaxis, :, :]
            #beta0,betaの計算
            Y = R @ self.X
            y_sqnorms = np.sum(Y**2,axis=1)[:,np.newaxis]
            q = g/g.sum()
            beta0 = 1.0 / np.sum( q * (self.x_sqnorms - y_sqnorms) )#(59)
            self.history['beta0'][e] = beta0
            #beta0 = 1.0 / (q.T @(np.diag(self.K) - np.diag(R@self.K@R.T)) )#(59)
            beta = beta0*g/(1+g)

            #E = np.sum((Y - self.X)**2,axis=1)[:,np.newaxis]#Kx1
            E = np.diag((R-I)@self.K@(R-I).T)[:,np.newaxis]#Kx1
            dB = 0.5 * (beta * E - 1)/(1 + g)
            dE = beta * (Phi - Phibar)
            A = R * ( dE + dB )
            dZ = ((A+A.T)[:,:,np.newaxis] * delta).sum(axis=1) - (self.alpha/self.gamma) * self.Z
            self.Z = self.Z + epsilon * dZ
            self.mat = dZ
            self.history['z'][e] = self.Z

            #M-step
            if self.L == 1:
                Zeta = np.linspace(self.Z.min(), self.Z.max(), self.M)
                Zeta = Zeta[:, np.newaxis]
                self.history['zeta'][e] = Zeta
                Dist = dist.cdist(Zeta, self.Z, 'sqeuclidean')
                H = np.exp(-0.5 * self.gamma*Dist)
                g = H.sum(axis=1)[:, np.newaxis]
                self.history['g'][e] = g.ravel()
                R = H / g
                self.history['f'][e] = R @ self.X
                V = R[:,:,np.newaxis]*(self.Z[np.newaxis,:,:] - Zeta[:,np.newaxis,:])#KxNxL
                Vbar = V.sum(axis=1)[:,np.newaxis,:]#Kx1xL
                dRdZ = V - R[:,:,np.newaxis]*Vbar#KxNxL
                dFdZ = np.sum(dRdZ[:,:,:,np.newaxis] * self.X[np.newaxis,:,np.newaxis,:],axis=1)#KxLxD
                delta_F = np.sum(dFdZ**2,axis=(1,2))#K dim vector
                self.history['fdelta'][e] = delta_F

            if self.L == 2:
                Zeta = np.meshgrid(np.linspace(self.Z[:, 0].min(), self.Z[:, 0].max(), math.sqrt(self.M)),
                                   np.linspace(self.Z[:, 1].min(), self.Z[:, 1].max(), math.sqrt(self.M)))
                Zeta = np.dstack(Zeta).reshape(self.M, self.L)
                self.history['zeta'][e] = Zeta
                Dist = dist.cdist(Zeta, self.Z, 'sqeuclidean')
                H = np.exp(-0.5 * self.gamma*Dist)
                g = H.sum(axis=1)[:, np.newaxis]
                self.history['g'][e] = g.ravel()
                R = H / g
                self.history['f'][e] = R @ self.X
                V = R[:,:,np.newaxis]*(self.Z[np.newaxis,:,:] - Zeta[:,np.newaxis,:])#KxNxL
                Vbar = V.sum(axis=1)[:,np.newaxis,:]#Kx1xL
                dRdZ = V - R[:,:,np.newaxis]*Vbar#KxNxL
                dFdZ = np.sum(dRdZ[:,:,:,np.newaxis] * self.X[np.newaxis,:,np.newaxis,:],axis=1)#KxLxD
                delta_F = np.sum(dFdZ**2,axis=(1,2))#K dim vector
                self.history['fdelta'][e] = delta_F
