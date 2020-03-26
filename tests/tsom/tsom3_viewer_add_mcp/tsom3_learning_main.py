
import numpy as np
from libs.models.tsom3 import TSOM3
from libs.models.tsom import TSOM2
from tsom3_viewer import TSOM3_Viewer

# X = np.loadtxt('datatsom.txt')
# X_re = np.reshape(X, (10, 10, 3, 1))

X = np.load('beverage_data.npy')
X_re = np.reshape(X, (604, 14, 11, 1))

np.random.seed(0)
# prepare init
latent_dim = np.array([2,2,2])
Z1init = np.random.rand(X_re.shape[0], latent_dim[0])
Z2init = np.random.rand(X_re.shape[1], latent_dim[1])
Z3init = np.random.rand(X_re.shape[2], latent_dim[2])
init = [Z1init, Z2init, Z3init]

tsom3 = TSOM3(X_re, latent_dim=2, resolution=10, SIGMA_MAX=3.0, SIGMA_MIN=0.2, TAU=50, init=init)
tsom3.fit(50)

# tsom2 = TSOM2(X, latent_dim=2, resolution=10, SIGMA_MAX=3.0, SIGMA_MIN=0.2, TAU=50)
# tsom2.fit(50)

np.save('tsom3_Y.npy',tsom3.Y)
np.save('tsom3_k1.npy',tsom3.k1_star)
np.save('tsom3_k2.npy',tsom3.k2_star)
np.save('tsom3_k3.npy',tsom3.k3_star)

# np.save('tsom2_Y.npy',tsom2.Y)
# np.save('tsom2_k1.npy',tsom2.k_star1)
# np.save('tsom2_k2.npy',tsom2.k_star2)

