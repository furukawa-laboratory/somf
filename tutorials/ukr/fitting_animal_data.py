from libs.models.unsupervised_kernel_regression import UnsupervisedKernelRegression as UKR
from libs.datasets.artificial import animal
from sklearn.utils import check_random_state

if __name__ == '__main__':
    n_components = 2
    bandwidth = 0.4
    lambda_ = 0.0
    is_compact = True
    is_save_history = True

    nb_epoch = 1000
    eta = 0.02

    X, labels_animal, labels_feature = animal.load_data(retlabel_animal=True,
                                                        retlabel_feature=True)

    seed = 14
    random_state = check_random_state(seed)
    init = random_state.normal(0.0, bandwidth * 0.1, size=(X.shape[0], n_components))

    ukr = UKR(X, n_components=n_components, bandwidth_gaussian_kernel=bandwidth,
              is_compact=is_compact, is_save_history=is_save_history, lambda_=lambda_, init=init)
    ukr.fit(nb_epoch=nb_epoch, eta=eta)

    params_imshow = {'cmap': 'BrBG',
                     'interpolation': 'spline16'}
    params_scatter = {'s': 30, 'marker': 'x'}
    ukr.visualize(
        n_grid_points=100,
        label_data=labels_animal,
        label_feature=labels_feature,
        is_middle_color_zero=True,
        is_show_all_label_data=True,
        params_imshow=params_imshow,
        params_scatter=params_scatter,
        title_latent_space='Animal map (latent space)',
        title_feature_bars='Animal feature'
    )
