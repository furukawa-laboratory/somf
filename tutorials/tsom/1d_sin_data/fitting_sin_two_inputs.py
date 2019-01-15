import numpy as np
from libs.datasets.artificial.sin_two_inputs import create_data
from libs.models.tsom import TSOM2
from libs.models.som import SOM

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _main():
    # fix random seed
    seed = 10
    np.random.seed(seed)

    # prepare observed data
    nb_samples1 = 30
    nb_samples2 = 30
    observed_dim = 1
    X_tsom, trueZ = create_data(nb_samples1, nb_samples2,
                                observed_dim=observed_dim, retz=True)

    X_som = X_tsom.reshape(nb_samples1 * nb_samples2, observed_dim)

    # set learning parameter
    nb_epoch = 25
    ## som
    latent_dim_som = 2
    resolution_som = 15
    SigmaMax_som = 2.0
    SigmaMin_som = 0.2
    Tau_som = 25

    ## tsom
    latent_dims_tsom = [1, 1]
    resolutions_tsom = [15, 15]
    SigmaMins_tsom = [0.2, 0.2]
    SigmaMaxs_tsom = [2.0, 2.0]
    Taus_tsom = [25, 25]

    # learn
    ## som
    som = SOM(X=X_som, latent_dim=latent_dim_som, resolution=resolution_som,
              sigma_max=SigmaMax_som, sigma_min=SigmaMin_som, tau=Tau_som, init='random')
    som.fit(nb_epoch=nb_epoch)
    ## tsom
    tsom = TSOM2(X=X_tsom, latent_dim=latent_dims_tsom, resolution=resolutions_tsom,
                 SIGMA_MAX=SigmaMaxs_tsom, SIGMA_MIN=SigmaMins_tsom, TAU=Taus_tsom, init='random')
    tsom.fit(nb_epoch=nb_epoch)

    # visualize learning process
    _visualize(X_tsom, trueZ, som, tsom)


def _visualize(X_tsom, trueZ, som, tsom):
    nb_epoch = som.history['y'].shape[0]
    observed_dim = som.history['y'].shape[2]
    resolution_som = int(np.sqrt(som.Zeta.shape[0]))
    latent_dim_som = som.Zeta.shape[1]

    ## prepare figure and axes
    fig = plt.figure(figsize=[15, 5])
    ax_data = fig.add_subplot(1, 3, 1, projection='3d')
    ax_som = fig.add_subplot(1, 3, 2, projection='3d')
    ax_tsom = fig.add_subplot(1, 3, 3, projection='3d')
    ax_data.set_title('observed data')

    ### set label
    ax_data.set_xlabel('true $z_1$')
    ax_data.set_ylabel('true $z_2$')
    ax_data.set_zlabel('x')

    ### setting to aspect equal
    max_range = 1.0
    mid_z1 = 0.0
    mid_z2 = 0.0
    mid_x1 = 0.0
    ax_data.set_aspect('equal')
    ax_data.set_xlim(mid_z1 - max_range, mid_z1 + max_range)
    ax_data.set_ylim(mid_z2 - max_range, mid_z2 + max_range)
    ax_data.set_zlim(mid_x1 - max_range, mid_x1 + max_range)

    ## draw observed data
    rgb_array = cmap2d(trueZ[:, :, 0].flatten(), trueZ[:, :, 1].flatten())
    ax_data.scatter(trueZ[:, :, 0].reshape(-1, 1),
                    trueZ[:, :, 1].reshape(-1, 1),
                    X_tsom[:, :, 0].reshape(-1, 1),
                    facecolors=rgb_array)

    ## prepare Zeta to draw
    Zeta_som_to_draw = som.Zeta.reshape(resolution_som, resolution_som, latent_dim_som)
    zeta1, zeta2 = np.meshgrid(tsom.Zeta2.flatten(), tsom.Zeta1.flatten())
    Zeta_tsom_to_draw = np.concatenate((zeta1[:, :, None], zeta2[:, :, None]), axis=2)

    ## define update function for funcanimation
    def update(epoch):
        ax_som.cla()
        ax_tsom.cla()
        ax_som.set_title('som')
        ax_tsom.set_title('tsom')
        fig.suptitle('compare som vs tsom epoch = {}'.format(epoch))
        for ax in [ax_som, ax_tsom]:
            ax.set_aspect('equal')
            ax.set_xlim(mid_z1 - max_range, mid_z1 + max_range)
            ax.set_ylim(mid_z2 - max_range, mid_z2 + max_range)
            ax.set_zlim(mid_x1 - max_range, mid_x1 + max_range)

        ### update som
        Y_som = som.history['y'][epoch]
        Z_som = som.history['z'][epoch]
        Y_som_to_draw = Y_som.reshape(resolution_som, resolution_som, observed_dim)
        ax_som.plot_wireframe(Zeta_som_to_draw[:, :, 0], Zeta_som_to_draw[:, :, 1], Y_som_to_draw[:, :, 0])
        ax_som.scatter(Z_som[:, 0], Z_som[:, 1], som.X[:, 0], facecolors=rgb_array)

        ax_som.set_xlabel('1st dim of 2d latent space')
        ax_som.set_ylabel('2nd dim of 2d latent space')
        ax_som.set_zlabel('1d observed space')

        ### update tsom
        Y_tsom = tsom.history['y'][epoch]
        Z1_now = tsom.history['z1'][epoch]
        Z2_now = tsom.history['z2'][epoch]
        z1, z2 = np.meshgrid(Z2_now.flatten(), Z1_now.flatten())
        Z_tsom = np.concatenate((z1[:, :, None], z2[:, :, None]), axis=2)

        ax_tsom.plot_wireframe(Zeta_tsom_to_draw[:, :, 0], Zeta_tsom_to_draw[:, :, 1], Y_tsom[:, :, 0])
        ax_tsom.scatter(Z_tsom[:, :, 0].flatten(), Z_tsom[:, :, 1].flatten(), tsom.X[:, :, 0].flatten(),
                        facecolors=rgb_array)

        ax_tsom.set_xlabel('latent space of 1st mode')
        ax_tsom.set_ylabel('latent space of 2nd mode')
        ax_tsom.set_zlabel('1d observed space')

    ani = animation.FuncAnimation(fig, update, interval=100, frames=nb_epoch, repeat=False)
    plt.show()


def cmap2d(z1, z2):
    length = z1.shape[0]
    z1_mms = (z1 - z1.min()) / (z1.max() - z1.min())
    z2_mms = (z2 - z2.min()) / (z2.max() - z2.min())
    # c_array = np.concatenate((z1.reshape(1,length),np.zeros((1,length)),z2.reshape(1,length)),axis=0)
    rgb_array = np.concatenate((z1_mms.reshape(length, 1), np.zeros((length, 1)), z2_mms.reshape(length, 1)), axis=1)
    # c_list = c_array.tolist()
    return rgb_array


if __name__ == '__main__':
    _main()
