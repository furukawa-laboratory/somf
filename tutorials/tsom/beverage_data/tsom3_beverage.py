from libs.datasets.real.beverage import load_data
from libs.models.tsom3 import TSOM3
import matplotlib.pyplot as plt





if __name__ == '__main__':
    # データのimport
    data_set = load_data(ret_situation_label=True, ret_beverage_label=True)
    X = data_set[0]
    beverage_label = data_set[1]
    situation_label = data_set[2]

    tsom3 = TSOM3(X, latent_dim=(2, 2, 2), resolution=(5, 5, 5), SIGMA_MAX=(0.1, 2.0, 2.0), SIGMA_MIN=(0.01, 0.2, 0.2),
                  TAU=(50, 50, 50), init='random')
    tsom3.fit(nb_epoch=250)

    # 結果の描画
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(tsom3.Z2[:, 0], tsom3.Z2[:, 1])
    for i in range(X.shape[1]):
        ax.text(tsom3.Z2[i, 0], tsom3.Z2[i, 1], beverage_label[i])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(tsom3.Z3[:, 0], tsom3.Z3[:, 1])
    for i in range(X.shape[2]):
        ax2.text(tsom3.Z3[i, 0], tsom3.Z3[i, 1], situation_label[i])
    plt.show()