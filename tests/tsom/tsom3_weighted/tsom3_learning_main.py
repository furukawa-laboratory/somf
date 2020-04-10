
import numpy as np
from tsom3_weighted import wTSOM3
from libs.models.tsom3 import TSOM3

X = np.loadtxt('datatsom.txt')
X = np.reshape(X, (10, 10, 3, 1))

np.random.seed(0)
# prepare init
latent_dim = np.array([2,2,2])
Z1init = np.random.rand(X.shape[0], latent_dim[0])
Z2init = np.random.rand(X.shape[1], latent_dim[1])
Z3init = np.random.rand(X.shape[2], latent_dim[2])
init = [Z1init, Z2init, Z3init]

wtsom3 = wTSOM3(X, latent_dim=2, resolution=5, SIGMA_MAX=2.0, SIGMA_MIN=0.2, TAU=50, init=init)
wtsom3.fit(10)

tsom3 = TSOM3(X, latent_dim=2, resolution=5, SIGMA_MAX=2.0, SIGMA_MIN=0.2, TAU=50, init=init)
tsom3.fit(10)

print(np.allclose(wtsom3.history['y'], tsom3.history['y']))

# np.save('tsom3_Y.npy',tsom3.Y)
# np.save('tsom3_k1.npy',tsom3.k1_star)
# np.save('tsom3_k2.npy',tsom3.k2_star)
# np.save('tsom3_k3.npy',tsom3.k3_star)
# np.save('wtsom3_Y.npy',wtsom3.Y)
# np.save('wtsom3_k1.npy',wtsom3.k_star1)
# np.save('wtsom3_k2.npy',wtsom3.k_star2)
# np.save('wtsom3_k3.npy',wtsom3.k_star3)

