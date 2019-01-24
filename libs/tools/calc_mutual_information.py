import numpy as np
import matplotlib.pyplot as plt


def calc_mutual_information(x, y, normalize=False, nb_bins=100, bias=0.01):
    # preprocessing and exception handling
    x_1d = np.squeeze(x)
    y_1d = np.squeeze(y)
    if x_1d.ndim != 1 or y_1d.ndim != 1:
        raise ValueError('now, this function follow latent_dim = 1 only.')
    if x_1d.shape != y_1d.shape:
        raise ValueError('x and y have to be same size')

    # make histogram
    hist_x, bin_edges = np.histogram(x_1d, bins=nb_bins)
    hist_y, bin_edges = np.histogram(y_1d, bins=nb_bins)
    hist_xy, x_edges, y_edges = np.histogram2d(x_1d, y_1d, bins=nb_bins)

    # make discrete probability distribution from histogram
    hist_x = hist_x.astype(np.float32) + bias
    hist_y = hist_y.astype(np.float32) + bias
    hist_xy = hist_xy.astype(np.float32) + bias
    prob_x = hist_x / hist_x.sum()
    prob_y = hist_y / hist_y.sum()
    prob_xy = hist_xy.astype(np.float32) / np.sum(hist_xy.astype(np.float32))

    # calculate mutual information
    MI = 0.0
    for k1 in range(nb_bins):
        for k2 in range(nb_bins):
            if prob_xy[k1][k2] != 0.0:
                MI += prob_xy[k1][k2] * np.log(prob_xy[k1][k2] / (prob_x[k1] * prob_y[k2]))
    # Using numpy, this calculation can be implemented by
    # MI = np.nan_to_num(prob_joint * np.log(prob_joint/(prob_true[:,None]*prob_estimate[None,:]))).sum()
    # However, if prob_joint[k1][k2] = 0, python interpreter warn.

    # if normalize is True, normalize mutual information to [0,1]
    if normalize:
        entropy_x = -np.sum(np.nan_to_num(prob_x * np.log(prob_x)))
        entropy_y = -np.sum(np.nan_to_num(prob_y * np.log(prob_y)))
        MI = MI / (0.5 * (entropy_x + entropy_y))

    return MI


if __name__ == '__main__':
    nb_samples = 10000
    seed = 100
    nb_bins = 100
    bias = 0.001

    np.random.seed(seed)

    x = np.linspace(-1.0, 1.0, nb_samples)
    y1 = x
    y2 = np.sqrt(x ** 2.0)
    y3 = np.random.rand(nb_samples)
    list_y = [y1, y2, y3]

    nb_patterns = len(list_y)
    fig = plt.figure(figsize=[3*nb_patterns, 4])
    for i, y in enumerate(list_y):
        MI = calc_mutual_information(x=x, y=y,
                                     nb_bins=nb_bins, bias=bias, normalize=False)
        NMI = calc_mutual_information(x=x, y=y,
                                      nb_bins=nb_bins, bias=bias, normalize=True)

        ax = fig.add_subplot(1, nb_patterns, i + 1)

        ax.hist2d(x.ravel(), y.ravel(), bins=nb_bins)
        ax.set_title("mutual information:{:.3f}\nnormalized one:{:.3f}".format(MI, NMI))
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')

    plt.show()
