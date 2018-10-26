import numpy as np

from libs.models.som import SOM
from libs.models.som_tensorflow import SOM as som
import time

N = 10000
D = 300
L = 2
resolution = 20
M = resolution ** L
seed = 100
np.random.seed(seed)
X = np.random.normal(0, 1, (N, D))
Zinit = np.random.rand(N,L)*2.0 -1.0

nb_epoch = 200
SIGMA_MAX = 2.2
SIGMA_MIN = 0.1
TAU = 50

start = time.time()
som_numpy = SOM(X, L, resolution, SIGMA_MAX, SIGMA_MIN, TAU, init=Zinit)
som_numpy.fit(nb_epoch=nb_epoch)
end = time.time()

print('Time for Numpy version with {0} input vectors of {1} dimensions and a resolution of {2} for reference vectors : {3}'.format(N, D, resolution, end - start))

start = time.time()
som_tensorflow = som(X.shape[1], X.shape[0], n=resolution, m=resolution,epochs=nb_epoch,sigma_max=SIGMA_MAX,sigma_min=SIGMA_MIN,tau=TAU,init=Zinit)
som_tensorflow.predict(X)
end = time.time()

print('Time for TensorFlow version with {0} input vectors of {1} dimensions and a resolution of {2} for reference vectors : {3}'.format(N, D, resolution, end-start))
