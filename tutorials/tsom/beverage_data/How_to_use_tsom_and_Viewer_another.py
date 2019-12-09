from libs.models.tsom import TSOM2
from libs.datasets.real.beverage import load_data
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V
import numpy as np


if __name__ == '__main__':
    X,beverage,situation=load_data(ret_beverage_label=True,ret_situation_label=True)

    np.random.seed(1)
    latent_dim = np.array([2, 2])
    Z1init = np.random.rand(X.shape[0], latent_dim[0])
    Z2init = np.random.rand(X.shape[1], latent_dim[1])
    init = [Z1init, Z2init]

    tsom = TSOM2(X,latent_dim=2,resolution=5,SIGMA_MAX=2.0,SIGMA_MIN=0.2,TAU=50)
    tsom.fit(nb_epoch=50)
    comp=TSOM2_V(y=tsom.Y,winner1=tsom.k_star1,winner2=tsom.k_star2,label1=None,label2=beverage,button_label=situation,init=init)
    comp.draw_map()
