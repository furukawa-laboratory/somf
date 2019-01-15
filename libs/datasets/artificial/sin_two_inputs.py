import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_data(nb_samples1, nb_samples2,
                observed_dim=1, retz=False):
    z1 = np.random.rand(nb_samples1) * 2.0 - 1.0
    z2 = np.random.rand(nb_samples2) * 2.0 - 1.0

    x = np.zeros((nb_samples1, nb_samples2, observed_dim))

    zz1, zz2 = np.meshgrid(z2, z1)
    observed_data = np.sin(0.75 * np.pi * (zz1 + zz2))

    x[:, :, 0] = observed_data
    z = np.concatenate((zz1[:, :, None], zz2[:, :, None]), axis=2)

    if retz:
        return x, z
    else:
        return x


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nb_samples1 = 25
    nb_samples2 = 20
    X, Z = create_data(nb_samples1=nb_samples1, nb_samples2=nb_samples2, observed_dim=1, retz=True)
    # ax.plot_wireframe(Z[:,:,0],Z[:,:,1],X[:,:,0])
    ax.scatter(Z[:, :, 0], Z[:, :, 1], X[:, :, 0])
    ax.set_xlabel('true $z_1$')
    ax.set_ylabel('true $z_2$')
    ax.set_zlabel('x')
    plt.show()
