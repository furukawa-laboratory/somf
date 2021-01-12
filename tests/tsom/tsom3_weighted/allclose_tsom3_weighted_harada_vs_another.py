#TSOM3のペアプロ用のファイル
import unittest
import numpy as np
from libs.models.tsom3 import TSOM3
from tests.tsom.tsom3_weighted.tsom3_weighted import wTSOM3
from tests.tsom.tsom3_weighted.tsom3_weighted_ishida import wTSOM3_ishida


# class TestTSOM3(unittest.TestCase):
# def test_harada_vs_another(self):

# random seed setting
seed = 100
np.random.seed(seed)

# prepare observed data
nb_samples1 = 5
nb_samples2 = 15
nb_samples3 = 20
nb_observed_dim=4

X = np.random.normal(0, 1, (nb_samples1, nb_samples2, nb_samples3, nb_observed_dim))

# set learning parameter
nb_epoch = 20
latent_dim = [2, 2, 2]
resolution = [5, 5, 5]
sigma_max = [2.0, 2.0, 2.0]
sigma_min = [0.2, 0.2, 0.2]
tau = [50, 50, 50]

## prepare init
Z1init = np.random.rand(X.shape[0], latent_dim[0])
Z2init = np.random.rand(X.shape[1], latent_dim[1])
Z3init = np.random.rand(X.shape[2], latent_dim[2])
init = [Z1init, Z2init, Z3init]

# generate tsom instance
tsom3_harada = wTSOM3(X, latent_dim=latent_dim, resolution=resolution,
SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau, init=init)

tsom3_ishida = wTSOM3_ishida(X, latent_dim=latent_dim, resolution=resolution,
SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau, init=init)

# learn
tsom3_harada.fit(nb_epoch=nb_epoch)

tsom3_ishida.fit(nb_epoch=nb_epoch)


# test
# np.testing.assert_allclose(tsom3_harada.H1, tsom3_ishida.H1, rtol=1e-4, atol=0)
print("h1", np.allclose(tsom3_harada.H1, tsom3_ishida.H1))
print("h2", np.allclose(tsom3_harada.H2, tsom3_ishida.H2))
print("h3", np.allclose(tsom3_harada.H3, tsom3_ishida.H3))
print("k1", np.allclose(tsom3_harada.k_star1, tsom3_ishida.k_star1))
print("k2", np.allclose(tsom3_harada.k_star2, tsom3_ishida.k_star2))
print("k3", np.allclose(tsom3_harada.k_star3, tsom3_ishida.k_star3))
# print("u1", np.allclose(tsom3_harada.U1, tsom3_ishida.U1))
# print("u2", np.allclose(tsom3_harada.U2, tsom3_ishida.U2))
# print("u3", np.allclose(tsom3_harada.U3, tsom3_ishida.U3))
print("y", np.allclose(tsom3_harada.Y, tsom3_ishida.Y))
print("yhist", np.allclose(tsom3_harada.history['y'], tsom3_ishida.history['y'], rtol=1e-4, atol=0))

np.testing.assert_allclose(tsom3_harada.history['y'], tsom3_ishida.history['y'], rtol=1e-4, atol=0)



# if __name__ == "__main__":
# unittest.main()
