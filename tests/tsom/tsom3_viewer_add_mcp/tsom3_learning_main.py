
import numpy as np
from libs.models.tsom3 import TSOM3
from tsom3_viewer import TSOM3_Viewer

X = np.loadtxt('datatsom.txt')
X = np.reshape(X, (10, 10, 3, 1))

np.random.seed(0)
# prepare init
latent_dim = np.array([2,2,2])
Z1init = np.random.rand(X.shape[0], latent_dim[0])
Z2init = np.random.rand(X.shape[1], latent_dim[1])
Z3init = np.random.rand(X.shape[2], latent_dim[2])
init = [Z1init, Z2init, Z3init]

tsom3 = TSOM3(X, latent_dim=2, resolution=10, SIGMA_MAX=3.0, SIGMA_MIN=0.2, TAU=50, init=init)
tsom3.fit(50)

np.save('tsom3_Y.npy',tsom3.Y)
np.save('tsom3_k1.npy',tsom3.k1_star)
np.save('tsom3_k2.npy',tsom3.k2_star)
np.save('tsom3_k3.npy',tsom3.k3_star)

