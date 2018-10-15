from libs.models.tsom import TSOM2
from libs.datasets.real.beverage import load_data
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V
import numpy as np

if __name__ == '__main__':
    X,beverage,situation=load_data(ret_beverage_label=True,ret_situation_label=True)

    tsom = TSOM2(X,latent_dim=2,resolution=10,SIGMA_MAX=2.0,SIGMA_MIN=0.2,TAU=50)
    tsom.fit(nb_epoch=3)
    comp=TSOM2_V(y=tsom.Y,winner1=tsom.k_star1,winner2=tsom.k_star2,label1=np.arange(X.shape[0]),label2=beverage,button_label=situation)
    comp.draw_map()