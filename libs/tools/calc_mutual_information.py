import numpy as np
import matplotlib.pyplot as plt


def calc_mutual_information(x, y, normalize=False, nb_bins=100, bias=0.01):
    x_1d = np.squeeze(x)
    y_1d = np.squeeze(y)
    if x_1d.ndim != 1 or y_1d.ndim != 1:
        raise ValueError('now, this function follow latent_dim = 1 only.')

    hist_x, bin_edges = np.histogram(x_1d, bins=nb_bins)
    hist_y, bin_edges = np.histogram(y_1d, bins=nb_bins)
    hist_xy, x_edges, y_edges = np.histogram2d(x_1d, y_1d, bins=nb_bins)

    hist_x = hist_x.astype(np.float32) + bias
    hist_y = hist_y.astype(np.float32) + bias
    hist_xy = hist_xy.astype(np.float32) + bias
    prob_x = hist_x / hist_x.sum()
    prob_y = hist_y / hist_y.sum()
    prob_xy = hist_xy.astype(np.float32) / np.sum(hist_xy.astype(np.float32))

    MI = 0.0
    for k1 in range(nb_bins):
        for k2 in range(nb_bins):
            if prob_xy[k1][k2] != 0.0:
                MI += prob_xy[k1][k2] * np.log(prob_xy[k1][k2] / (prob_x[k1] * prob_y[k2]))
    # Using numpy, this calculation can be implemented by
    # MI = np.nan_to_num(prob_joint * np.log(prob_joint/(prob_true[:,None]*prob_estimate[None,:]))).sum()
    # However, if prob_joint[k1][k2] = 0, python interpreter warn.

    if normalize:
        entropy_x = -np.sum(np.nan_to_num(prob_x * np.log(prob_x)))
        entropy_y = -np.sum(np.nan_to_num(prob_y * np.log(prob_y)))
        MI = MI / (0.5 * (entropy_x + entropy_y))
    else:
        pass

    return MI


if __name__ == '__main__':
    nb_samples = 10000
    seed = 100
    nb_bins = 10
    bias = 0.001

    np.random.rand(seed)

    Z_true = np.linspace(-1.0, 1.0, nb_samples)[:, None]
    Z_estimate1 = Z_true  # + 0.01 * (np.random.rand(nb_samples)[:,None] * 2.0-1.0)
    Z_estimate2 = np.sqrt(Z_true ** 2.0)  # + 0.01 * (np.random.rand(nb_samples)[:,None] * 2.0-1.0)
    Z_estimate3 = np.random.rand(nb_samples, 1)
    list_Z_estimate = [Z_estimate1, Z_estimate2, Z_estimate3]

    fig = plt.figure(figsize=[9, 4])
    nb_patterns = len(list_Z_estimate)
    for i, Z_estimate in enumerate(list_Z_estimate):
        MI = calc_mutual_information(x=Z_true, y=Z_estimate,
                                     nb_bins=nb_bins, bias=bias, normalize=False)
        NMI = calc_mutual_information(x=Z_true, y=Z_estimate,
                                      nb_bins=nb_bins, bias=bias, normalize=True)

        ax = fig.add_subplot(1, nb_patterns, i + 1)

        ax.hist2d(Z_true.ravel(), Z_estimate.ravel(), bins=nb_bins)
        ax.set_title("mutual information:{:.3f}\nnormalized one:{:.3f}".format(MI, NMI))
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')

    plt.show()
