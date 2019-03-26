import numpy as np

from libs.models.tsom import TSOM2
from libs.models.tsom_tensorflow import TSOM2 as tsom
import time

nb_samples1 = 50
nb_samples2 = 50
observed_dim = 10

X = np.random.normal(0, 1, (nb_samples1, nb_samples2, observed_dim))

L = 2
resolution = [20, 20]
latent_dim = [2, 2]
seed = 100
np.random.seed(seed)

Z1init = np.random.rand(X.shape[0], latent_dim[0])
Z2init = np.random.rand(X.shape[1], latent_dim[1])
init = [Z1init, Z2init]

nb_epoch = 200
SIGMA_MAX = [2.2, 2.0]
SIGMA_MIN = [0.4, 0.2]
TAU = [60, 50]

start = time.time()
som_numpy = TSOM2(X, latent_dim=latent_dim, resolution=resolution, SIGMA_MAX=SIGMA_MAX, SIGMA_MIN=SIGMA_MIN, TAU=TAU, init=init)
som_numpy.fit(nb_epoch=nb_epoch)
end = time.time()

print('Time for Numpy version with {0} and {1} input vectors of {2} dimensions and a resolution of {3} for reference vectors : {4}'.format(nb_samples1, nb_samples2, observed_dim, resolution, end - start))

start = time.time()
som_tensorflow = tsom(X.shape[2], [X.shape[0], X.shape[1]], n=resolution, m=resolution,epochs=nb_epoch,sigma_max=SIGMA_MAX,sigma_min=SIGMA_MIN,tau=TAU,init=init)
som_tensorflow.predict(X)
end = time.time()

print('Time for TensorFlow version with {0} and {1} input vectors of {2} dimensions and a resolution of {3} for reference vectors : {4}'.format(nb_samples1, nb_samples2, observed_dim, resolution, end - start))
