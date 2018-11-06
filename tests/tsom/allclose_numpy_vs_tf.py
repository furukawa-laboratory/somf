import unittest

import numpy as np

from libs.datasets.artificial.animal import load_data
from libs.models.tsom import TSOM2
from libs.models.tsom_tensorflow import TSOM2 as ts
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V


class TestTSOM2(unittest.TestCase):
    def test_numpy_vs_tf(self):
        # random seed setting
        seed = 100
        np.random.seed(seed)

        # prepare observed data
        nb_samples1 = 10
        nb_samples2 = 5
        observed_dim = 2

        X = np.random.normal(0, 1, (nb_samples1, nb_samples2, observed_dim))





        # set learning parameter
        nb_epoch = 60
        latent_dim = [2, 2]
        resolution = [10, 10]
        sigma_max = [2.0, 2.0]
        sigma_min = [0.2, 0.2]
        tau = [50, 50]

        ## prepare init
        Z1init = np.random.rand(X.shape[0], latent_dim[0])
        Z2init = np.random.rand(X.shape[1], latent_dim[1])
        init = [Z1init, Z2init]

        # generate tsom instance
        tsom_numpy = TSOM2(X, latent_dim=latent_dim, resolution=resolution,
                           SIGMA_MAX=sigma_max, SIGMA_MIN=sigma_min, TAU=tau,
                           init=init)
        tsom_numpy.fit(nb_epoch=nb_epoch)




        tsom_tensorflow = ts(X.shape[2], [X.shape[0], X.shape[1]] ,epochs=nb_epoch, n=[resolution[0], resolution[1]],
                             m=[resolution[0], resolution[1]], sigma_min=sigma_min, sigma_max=sigma_max, tau=tau, init=init)

        # learn

        tsom_tensorflow.predict(X)



        # test
        np.testing.assert_allclose(tsom_numpy.history['y'], tsom_tensorflow.historyY)






if __name__ == "__main__":
    unittest.main()
