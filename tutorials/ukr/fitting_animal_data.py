import numpy as np

from libs.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as UKR
from libs.datasets.artificial import animal

if __name__ == '__main__':
    n_components = 2
    bandwidth = 0.5
    lambda_ = 0.0
    is_compact = True
    is_save_history = True

    nb_epoch = 100
    eta = 5.0

    resolution = 100

    X, labels_animal, labels_feature = animal.load_data(retlabel_animal=True,
                                                        retlabel_feature=True)

    seed = 13
    np.random.seed(seed)

    ukr = UKR(X, n_components=n_components, bandwidth_gaussian_kernel=bandwidth,
              is_compact=is_compact, is_save_history=is_save_history, lambda_=lambda_)
    ukr.fit(nb_epoch=nb_epoch, eta=eta)

    ukr.visualize(n_representative_points=resolution,
                  label_data = labels_animal,
                  label_feature=labels_feature)
