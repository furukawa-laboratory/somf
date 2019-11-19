import numpy as np
import scipy.spatial.distance as dist
from tqdm import tqdm


class UnsupervisedKernelRegression(object):
    def __init__(self, X, latent_dim,sigma=0.2,
                 isGEM=False,
                 isCompact=True,alpha=0.0,
                 init='random',isLOOCV=False):
        self.X = X.copy()
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.L = latent_dim
        self.sigma = sigma
        self.gamma = 1.0 / (sigma*sigma)
        self.isCompact = isCompact
        self.isGEM = isGEM
        self.isLOOCV = isLOOCV

        self.Z = None
        if isinstance(init, str) and init in 'random':
            self.Z = np.random.normal(0, 0.1, (self.N, self.L))
        elif isinstance(init, np.ndarray) and init.shape == (self.N, self.L):
            self.Z = init.copy()
        else:
            raise ValueError("invalid init: {}".format(init))

        self.Alpha = alpha

        self.history = {}

        self.donefit = False



    def fit(self, nb_epoch=100, verbose=True, eta=0.5, expand_epoch=None):

        K = self.X @ self.X.T
        X2 = np.diag(K)[:, None]
        # Xn = np.sum(np.square(self.X[:, None, :] - self.X[None, :, :]), axis=2)
        # DistX = Xn.reshape(self.N**2, 1)

        self.nb_epoch = nb_epoch

        self.history['z'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y'] = np.zeros((nb_epoch, self.N, self.D))
        self.history['zvar'] = np.zeros((nb_epoch, self.L))
        self.history['obj_func'] = np.zeros(nb_epoch)


        if verbose:
            bar = tqdm(range(nb_epoch))
        else:
            bar = range(nb_epoch)


        for epoch in bar:
            Delta = self.Z[:, None, :] - self.Z[None, :, :]
            DistZ = np.sum(np.square(Delta), axis=2)
            H = np.exp(-0.5 * self.gamma * DistZ)
            if self.isLOOCV:
                H -= np.identity(H.shape[0])

            # Hprime = H
            G = np.sum(H, axis=1)[:, None]
            GInv = 1 / G
            R = H * GInv
            # Rprime = Hprime * GInv

            Y = R @ self.X
            # Y2 = np.sum(np.square(Y), axis=1)[:, None]
            # beta0 = np.sum(G) / np.sum(G * (X2 - Y2))
            DeltaYX = Y[:,None,:] - self.X[None, :, :]
            Error = Y - self.X
            obj_func = -0.5 * np.sum(np.square(Error)) - 0.5*self.Alpha*np.sum(np.square(self.Z))


            A = self.gamma * R * np.einsum('nd,nid->ni', Y - self.X, DeltaYX)
            #dFdZ = -beta0 * np.sum((A + A.T)[:, :, None] * Delta, axis=1)
            if self.isGEM:
                dFdZ = -np.sum(A[:, :, None] * Delta, axis=1)
            else:
                dFdZ = -np.sum((A + A.T)[:, :, None] * Delta, axis=1)

            dFdZ -= self.Alpha * self.Z

            # self.Z += (eta / self.D) * dFdZ
            self.Z += eta * dFdZ
            if self.isCompact:
                self.Z = np.clip(self.Z,-1.0,1.0)
            else:
                self.Z -= self.Z.mean(axis=0)


            self.history['z'][epoch] = self.Z
            self.history['y'][epoch] = Y
            self.history['zvar'][epoch] = np.mean(np.square(self.Z - self.Z.mean(axis=0)),axis=0)
            self.history['obj_func'][epoch] = obj_func



        self.donefit = True
        return self.history

    def calcF(self, resolution, size='auto'):
        """
        :param resolution:
        :param size:
        :return:
        """
        if not self.donefit:
            raise ValueError("fit is not done")

        self.resolution = resolution
        Zeta = create_zeta(-1, 1, self.L, resolution)
        M = Zeta.shape[0]

        self.history['f'] = np.zeros((self.nb_epoch, M, self.D))

        for epoch in range(self.nb_epoch):
            Z = self.history['z'][epoch]
            if size == 'auto':
                Zeta = create_zeta(Z.min(), Z.max(), self.L, resolution)
            else:
                Zeta = create_zeta(size.min(), size.max(), self.L, resolution)

            Dist = dist.cdist(Zeta, Z, 'sqeuclidean')

            H = np.exp(-0.5 *self.gamma* Dist)
            G = np.sum(H, axis=1)[:, None]
            GInv = np.reciprocal(G)
            R = H * GInv

            Y = np.dot(R, self.X)

            self.history['f'][epoch] = Y

    def transform(self, Xnew, nb_epoch_trans=100, eta_trans=0.5, verbose=True, constrained=True):
        # calculate latent variables of new data using gradient descent
        # objective function is square error E = ||f(z)-x||^2

        if not self.donefit:
            raise ValueError("fit is not done")

        Nnew = Xnew.shape[0]

        # initialize Znew, using latent variables of observed data
        Dist_Xnew_X = dist.cdist(Xnew, self.X)
        BMS = np.argmin(Dist_Xnew_X, axis=1) # calculate Best Matching Sample
        Znew = self.Z[BMS,:] # initialize Znew

        if verbose:
            bar = tqdm(range(nb_epoch_trans))
        else:
            bar = range(nb_epoch_trans)

        for epoch in bar:
            # calculate gradient
            Delta = self.Z[None,:,:] - Znew[:,None,:]                   # shape = (Nnew,N,L)
            Dist_Znew_Z = dist.cdist(Znew,self.Z,"sqeuclidean")         # shape = (Nnew,N)
            H = np.exp(-0.5 *self.gamma* Dist_Znew_Z)                              # shape = (Nnew,N)
            G = np.sum(H,axis=1)[:,None]                                # shape = (Nnew,1)
            Ginv = np.reciprocal(G)                                     # shape = (Nnew,1)
            R = H * Ginv                                                # shape = (Nnew,N)
            F = R @ self.X                                              # shape = (Nnew,D)

            Delta_bar = np.einsum("kn,knl->kl",R,Delta)                 # (Nnew,N)times(Nnew,N,L)=(Nnew,L)
            # Delta_bar = np.sum(R[:,:,None] * Delta, axis=1)           # same calculate
            dRdZ = self.gamma * R[:,:,None] * (Delta - Delta_bar[:,None,:])          # shape = (Nnew,N,L)

            dFdZ = np.einsum("nd,knl->kdl",self.X,dRdZ)                 # shape = (Nnew,D,L)
            # dFdZ = np.sum(self.X[None,:,:,None]*dRdZ[:,:,None,:],axis=1)  # same calculate
            dEdZ = 2.0 * np.einsum("kd,kdl->kl",F-Xnew,dFdZ) # shape (Nnew, L)
            # update latent variables
            Znew -= eta_trans * dEdZ
            if self.isCompact:
                Znew = np.clip(Znew,-1.0,1.0)
            if constrained:
                Znew = np.clip(Znew, self.Z.min(axis=0), self.Z.max(axis=0))

        return Znew

    def inverse_transform(self, Znew):
        if not self.donefit:
            raise ValueError("fit is not done")
        if Znew.shape[1]!=self.L:
            raise ValueError("Znew dimension must be {}".format(self.L))

        Dist_Znew_Z = dist.cdist(Znew,self.Z,"sqeuclidean")         # shape = (Nnew,N)
        H = np.exp(-0.5 * self.gamma *Dist_Znew_Z)                              # shape = (Nnew,N)
        G = np.sum(H,axis=1)[:,None]                                # shape = (Nnew,1)
        Ginv = np.reciprocal(G)                                     # shape = (Nnew,1)
        R = H * Ginv                                                # shape = (Nnew,N)
        F = R @ self.X                                              # shape = (Nnew,D)

        return F




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
