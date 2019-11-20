import numpy as np
from libs.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as UKR
from libs.models.som import SOM
from libs.visualization.som.animation_learning_process_3d import anime_learning_process_3d
from libs.datasets.artificial.kura import create_data
import matplotlib

if __name__ == '__main__':

    # create artiricial dataset
    nb_samples = 500
    seed = 1
    np.random.seed(seed)
    X = create_data(nb_samples)
    x_sigma = 0.1
    X += np.random.normal(0,x_sigma,X.shape)

    # common parameter
    n_components = 2
    bandwidth_gaussian_kernel = 0.2
    nb_epoch = 100

    # ukr parameter
    is_compact = True
    is_save_history = True
    lambda_ = 0.0
    eta = 0.02

    # som parameter
    tau = nb_epoch
    init_bandwidth = 2.0
    resolution = 10


    som = SOM(X, latent_dim=n_components, resolution=resolution,
              sigma_max=init_bandwidth, sigma_min=bandwidth_gaussian_kernel, tau=tau)
    ukr = UKR(X, n_components=n_components,bandwidth_gaussian_kernel=bandwidth_gaussian_kernel,
              is_compact=is_compact,is_save_history=is_save_history,lambda_=lambda_)
    som.fit(nb_epoch=nb_epoch)
    ukr.fit(nb_epoch=nb_epoch,eta=eta)
    ukr.calculation_history_of_mapping(resolution=30)

    anime_learning_process_3d(X=ukr.X, Y_allepoch=ukr.history['f'])
