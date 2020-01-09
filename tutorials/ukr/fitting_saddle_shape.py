import numpy as np
from libs.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as UKR
from libs.models.som import SOM
from libs.datasets.artificial.kura import create_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

if __name__ == '__main__':
    # create artiricial dataset
    nb_samples = 500
    seed = 1
    np.random.seed(seed)
    X = create_data(nb_samples)
    x_sigma = 0.1
    X += np.random.normal(0, x_sigma, X.shape)

    # common parameter
    n_components = 2
    bandwidth_gaussian_kernel = 0.2
    nb_epoch = 100

    # ukr parameter
    is_compact = True
    is_save_history = True
    lambda_ = 0.0
    eta = 8.0

    # som parameter
    tau = nb_epoch
    init_bandwidth = 2.0
    resolution = 10

    # learn ukr and som
    som = SOM(X, latent_dim=n_components, resolution=resolution,
              sigma_max=init_bandwidth, sigma_min=bandwidth_gaussian_kernel, tau=tau)
    ukr = UKR(X, n_components=n_components, bandwidth_gaussian_kernel=bandwidth_gaussian_kernel,
              is_compact=is_compact, is_save_history=is_save_history, lambda_=lambda_)
    som.fit(nb_epoch=nb_epoch)
    ukr.fit(nb_epoch=nb_epoch, eta=eta)
    ukr.calculation_history_of_mapping(resolution=30)

    fig = plt.figure(figsize=[7, 8])
    ax_latent_space_som = fig.add_subplot(2, 2, 1, aspect='equal')
    ax_data_space_som = fig.add_subplot(2, 2, 2, aspect='equal', projection='3d')
    ax_latent_space_ukr = fig.add_subplot(2, 2, 3, aspect='equal')
    ax_data_space_ukr = fig.add_subplot(2, 2, 4, aspect='equal', projection='3d')


    def plot(i):
        ax_latent_space_som.cla()
        ax_data_space_som.cla()
        ax_data_space_ukr.cla()
        ax_latent_space_ukr.cla()
        ax_data_space_som.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, c=X[:, 0], alpha=0.5)
        ax_data_space_ukr.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, c=X[:, 0], alpha=0.5)
        mapping_2d_som = som.history['y'][i].reshape(resolution, resolution, X.shape[1])
        ax_data_space_som.plot_wireframe(mapping_2d_som[:, :, 0],
                                         mapping_2d_som[:, :, 1],
                                         mapping_2d_som[:, :, 2])
        mapping_2d_ukr = ukr.history['f'][i].reshape(30, 30, X.shape[1])
        ax_data_space_ukr.plot_surface(mapping_2d_ukr[:, :, 0],
                                       mapping_2d_ukr[:, :, 1],
                                       mapping_2d_ukr[:, :, 2],
                                       antialiased=False)
        ith_z_som = som.history['z'][i]
        ith_z_ukr = ukr.history['z'][i]
        ax_latent_space_som.scatter(ith_z_som[:, 0], ith_z_som[:, 1], s=3, c=X[:, 0])
        ax_latent_space_ukr.scatter(ith_z_ukr[:, 0], ith_z_ukr[:, 1], s=3, c=X[:, 0])
        fig.suptitle("epoch {}".format(i))

        ax_latent_space_som.set_xlim(-1.0, 1.0)
        ax_latent_space_som.set_ylim(-1.0, 1.0)
        ax_latent_space_ukr.set_xlim(-1.0, 1.0)
        ax_latent_space_ukr.set_ylim(-1.0, 1.0)

        ax_latent_space_som.set_title('som latent space')
        ax_latent_space_ukr.set_title('ukr latent space')
        ax_data_space_som.set_title('som data space')
        ax_data_space_ukr.set_title('ukr data space')


    ani = animation.FuncAnimation(fig, plot, frames=nb_epoch, interval=20, repeat=False)
    plt.show()

    # anime_learning_process_3d(X=ukr.X, Y_allepoch=ukr.history['f'])
#
