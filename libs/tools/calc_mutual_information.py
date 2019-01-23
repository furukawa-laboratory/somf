import numpy as np
import matplotlib.pyplot as plt

def calc_mutual_information(x, y, nb_bins=100, bias=0.0001):
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError('now, this function follow latent_dim = 1 only')

    hist_true,bin_edges = np.histogram(x.ravel(), bins=nb_bins)
    hist_estimate,bin_edges = np.histogram(y.ravel(), bins=nb_bins)
    hist_joint,x_edges,y_edges = np.histogram2d(x.ravel(), y.ravel(), bins=nb_bins)

    hist_true = hist_true.astype(np.float32) + bias
    hist_estimate = hist_estimate.astype(np.float32) + bias
    prob_true = hist_true / hist_true.sum()
    prob_estimate = hist_estimate / hist_estimate.sum()
    prob_joint = hist_joint.astype(np.float32) / np.sum(hist_joint.astype(np.float32))


    MI_numpy = 0.0
    for k1 in range(nb_bins):
        for k2 in range(nb_bins):
            if prob_joint[k1][k2] != 0:
                MI_numpy += prob_joint[k1][k2] * np.log(prob_joint[k1][k2]/(prob_true[k1] * prob_estimate[k2]))



    return MI_numpy


if __name__ == '__main__':
    nb_samples = 5000
    seed = 100
    nb_bins = 20

    np.random.rand(seed)

    Z_true = np.linspace(-1.0,1.0,nb_samples)[:,None]
    Z_estimate1 = Z_true #+ 0.01 * (np.random.rand(nb_samples)[:,None] * 2.0-1.0)
    Z_estimate2 = np.sqrt(Z_true**2.0) #+ 0.01 * (np.random.rand(nb_samples)[:,None] * 2.0-1.0)
    Z_estimate3 = np.random.rand(nb_samples,1)
    list_Z_estimate = [Z_estimate1,Z_estimate2,Z_estimate3]

    fig = plt.figure(figsize=[9,3])
    nb_patterns = len(list_Z_estimate)
    for i,Z_estimate in enumerate(list_Z_estimate):
        MI = calc_mutual_information(x=Z_true, y=Z_estimate, nb_bins=nb_bins)

        ax = fig.add_subplot(1,nb_patterns,i+1)

        ax.hist2d(Z_true.ravel(), Z_estimate.ravel(), bins=nb_bins)
        ax.set_title("MI:{}".format(MI))


    plt.show()



