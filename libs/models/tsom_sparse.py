import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from tqdm import tqdm
from ..tools.create_zeta import create_zeta


class TSOM2():
    def __init__(self, x, latent_dim, resolution, sigma_max, sigma_min, tau, init='random'):
        # validation
        if not sparse.isspmatrix(x):
            raise TypeError("type(x) must be sparse")
        if not x.ndim == 2:
            raise ValueError("x.ndim must be 2")

        for param in [sigma_max, sigma_min, tau]:
            if (type(param) is not float) and (not isinstance(param, (list, tuple))):
                raise TypeError(f"invalid type: {param}")

        for param in [resolution, latent_dim]:
            if (type(param) is not int) and (not isinstance(param, (list, tuple))):
                raise TypeError(f"invalid type: {param}")

        # initialization
        self.x = x.copy()
        self.x_nonzero = x.copy()
        self.x_nonzero.data = np.ones(x.size)
        (self.n_samples1, self.n_samples2) = self.x.shape

        if type(sigma_max) is float:
            self.sigma_max1 = sigma_max
            self.sigma_max2 = sigma_max
        elif isinstance(sigma_max, (list, tuple)):
            self.sigma_max1 = sigma_max[0]
            self.sigma_max2 = sigma_max[1]

        if type(sigma_min) is float:
            self.sigma_min1 = sigma_min
            self.sigma_min2 = sigma_min
        elif isinstance(sigma_min, (list, tuple)):
            self.sigma_min1 = sigma_min[0]
            self.sigma_min2 = sigma_min[1]

        if type(tau) is float:
            self.tau1 = tau
            self.tau2 = tau
        elif isinstance(tau, (list, tuple)):
            self.tau1 = tau[0]
            self.tau2 = tau[1]

        if type(resolution) is int:
            self.resolution1 = resolution
            self.resolution2 = resolution
        elif isinstance(resolution, (list, tuple)):
            self.resolution1 = resolution[0]
            self.resolution2 = resolution[1]

        if type(latent_dim) is int:
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim
        elif isinstance(latent_dim, (list, tuple)):
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]

        self.zeta1 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim1, resolution=self.resolution1, include_min_max=True)
        self.zeta2 = create_zeta(-1.0, 1.0, latent_dim=self.latent_dim2, resolution=self.resolution2, include_min_max=True)

        self.n_units1 = self.zeta1.shape[0]
        self.n_units2 = self.zeta2.shape[0]

        self.k_star1 = None
        self.k_star2 = None
        self.z1 = None
        self.z2 = None
        self.h1 = None
        self.h2 = None
        self.u = None
        self.v = None
        self.y = None

        self.init = init
        self.history = {}

    def initialization(self, nb_epoch, is_direct):
        if isinstance(self.init, str) and self.init in 'random':
            self.z1 = np.random.rand(self.n_samples1, self.latent_dim1) * 2.0 - 1.0
            self.z2 = np.random.rand(self.n_samples2, self.latent_dim2) * 2.0 - 1.0
        elif isinstance(self.init, (tuple, list)) and len(self.init) == 2:
            if isinstance(self.init[0], np.ndarray) and self.init[0].shape == (self.n_samples1, self.latent_dim1):
                self.z1 = self.init[0].copy()
            else:
                raise ValueError("invalid inits[0]: {}".format(self.init))
            if isinstance(self.init[1], np.ndarray) and self.init[1].shape == (self.n_samples2, self.latent_dim2):
                self.z2 = self.init[1].copy()
            else:
                raise ValueError("invalid inits[1]: {}".format(self.init))
        else:
            raise ValueError("invalid inits: {}".format(self.init))

        self.history['y'] = np.zeros((nb_epoch, self.n_units1, self.n_units2))
        self.history['z1'] = np.zeros((nb_epoch, self.n_samples1, self.latent_dim1))
        self.history['z2'] = np.zeros((nb_epoch, self.n_samples2, self.latent_dim2))

        self.m_step(0, is_direct)

    def fit(self, nb_epoch=200, is_direct=True):
        self.initialization(nb_epoch, is_direct)

        for epoch in tqdm(np.arange(nb_epoch)):
            self.e_step(is_direct)
            self.m_step(epoch, is_direct)

            self.history['y'][epoch, :, :] = self.y
            self.history['z1'][epoch, :] = self.z1
            self.history['z2'][epoch, :] = self.z2

    def e_step(self, is_direct):
        if is_direct:
            self.e_step_direct()
        else:
            self.e_step_indirect()

    def m_step(self, epoch, is_direct):
        if is_direct:
            self.m_step_direct(epoch)
        else:
            self.m_step_indirect(epoch)

    def e_step_indirect(self):
        self.k_star1 = np.argmin(np.sum(np.square(self.u[:, None, :, :] - self.y[None, :, :, :]), axis=(2, 3)), axis=1)
        self.k_star2 = np.argmin(np.sum(np.square(self.v[:, :, None, :] - self.y[:, None, :, :]), axis=(0, 3)), axis=1)
        self.z1 = self.zeta1[self.k_star1, :]
        self.z2 = self.zeta2[self.k_star2, :]

    def m_step_indirect(self, epoch):
        sigma1 = self.sigma_min1 + (self.sigma_max1 - self.sigma_min1) * np.exp(-epoch / self.tau1)
        h1 = np.exp(-cdist(self.zeta1, self.z1, 'sqeuclidean') / (2 * pow(sigma1, 2)))
        g1 = np.sum(h1, axis=1, keepdims=True)
        r1 = h1 / g1

        sigma2 = self.sigma_min2 + (self.sigma_max2 - self.sigma_min2) * np.exp(-epoch / self.tau2)
        h2 = np.exp(-cdist(self.zeta2, self.z2, 'sqeuclidean') / (2 * pow(sigma2, 2)))
        g2 = np.sum(h2, axis=1, keepdims=True)
        r2 = h2 / g2

        self.u = self.x @ r2.T
        self.v = r1 @ self.x
        self.y = r1 @ self.u

    def e_step_direct(self):
        self.k_star1 = np.argmin(
            np.sum(self.y ** 2 @ self.h2, axis=1, keepdims=True) - 2 * self.y @ self.h2 @ self.x.T,
            axis=0,
        )
        self.k_star2 = np.argmin(
            np.sum(self.y.T ** 2 @ self.h1, axis=1, keepdims=True) - 2 * self.y.T @ self.h1 @ self.x,
            axis=0,
        )
        self.z1 = self.zeta1[self.k_star1, :]
        self.z2 = self.zeta2[self.k_star2, :]

    def m_step_direct(self, epoch):
        sigma1 = self.sigma_min1 + (self.sigma_max1 - self.sigma_min1) * np.exp(-epoch / self.tau1)
        self.h1 = np.exp(-cdist(self.zeta1, self.z1, 'sqeuclidean') / (2 * pow(sigma1, 2)))
        g1 = np.sum(self.h1, axis=1, keepdims=True)
        r1 = self.h1 / g1

        sigma2 = self.sigma_min2 + (self.sigma_max2 - self.sigma_min2) * np.exp(-epoch / self.tau2)
        self.h2 = np.exp(-cdist(self.zeta2, self.z2, 'sqeuclidean') / (2 * pow(sigma2, 2)))
        g2 = np.sum(self.h2, axis=1, keepdims=True)
        r2 = self.h2 / g2

        self.y = r1 @ self.x @ r2.T
