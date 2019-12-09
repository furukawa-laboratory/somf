from libs.datasets.real.beverage import load_data
from libs.models.tsom3 import TSOM3
import matplotlib.pyplot as plt
from libs.visualization.tsom.tsom3_viewer import TSOM3_Viewer as TSOM3_V
import numpy as np


if __name__ == '__main__':
    # データのimport
    data_set = load_data(ret_situation_label=True, ret_beverage_label=True)
    X = data_set[0]
    beverage_label = data_set[1]
    situation_label = data_set[2]

    np.random.seed(1)
    latent_dim = np.array([2, 2])
    Z1init = np.random.rand(X.shape[0], latent_dim[0])
    Z2init = np.random.rand(X.shape[1], latent_dim[1])
    init = [Z1init, Z2init]

    tsom3 = TSOM3(X, latent_dim=2, resolution=5, SIGMA_MAX=2.0, SIGMA_MIN=0.2,
                  TAU=(50, 50, 50), init="random")
    tsom3.fit(nb_epoch=250)

    # 結果の描画
    comp = TSOM3_V(y=tsom3.Y, winner1=tsom3.k1_star, winner2=tsom3.k2_star, winner3=tsom3.k3_star, label1=None, label2=beverage_label, label3=situation_label, init=init)
    comp.draw_map()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # ax.scatter(tsom3.Z2[:, 0], tsom3.Z2[:, 1])
    # for i in range(X.shape[1]):
    #     ax.text(tsom3.Z2[i, 0], tsom3.Z2[i, 1], beverage_label[i])
    #
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.scatter(tsom3.Z3[:, 0], tsom3.Z3[:, 1])
    # for i in range(X.shape[2]):
    #     ax2.text(tsom3.Z3[i, 0], tsom3.Z3[i, 1], situation_label[i])
    # plt.show()