import numpy as np


#
def load_kura_tsom(xsamples, ysamples, retz=False):
    z1 = np.linspace(-1, 1, xsamples)
    z2 = np.linspace(-1, 1, ysamples)

    z1_repeated, z2_repeated = np.meshgrid(z1, z2, indexing='ij')
    x1 = z1_repeated
    x2 = z2_repeated
    x3 = z1_repeated ** 2.0 - z2_repeated ** 2.0

    x = np.concatenate((x1[:, :, np.newaxis], x2[:, :, np.newaxis], x3[:, :, np.newaxis]), axis=2)
    truez = np.concatenate((z1_repeated[:, :, np.newaxis], z2_repeated[:, :, np.newaxis]), axis=2)

    if retz:
        return x, truez
    else:
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 10
    ysamples = 10

    x, truez = load_kura_tsom(10, 10, retz=True)

    fig = plt.figure(figsize=[10, 5])
    ax_x = fig.add_subplot(1, 2, 1, projection='3d')
    ax_truez = fig.add_subplot(1, 2, 2)
    ax_x.scatter(x[:, :, 0].flatten(), x[:, :, 1].flatten(), x[:, :, 2].flatten(), c=x[:, :, 0].flatten())
    ax_truez.scatter(truez[:, :, 0].flatten(), truez[:, :, 1].flatten(), c=x[:, :, 0].flatten())
    ax_x.set_title('Generated three-dimensional data')
    ax_truez.set_title('True two-dimensional latent variable')
    plt.show()
