from libs.models.tsom import TSOM2
from libs.datasets.artificial.animal import load_data
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V
import numpy as np

if __name__ == '__main__':
    X,labels_animal,labels_feature=load_data(retlabel_animal=True,retlabel_feature=True)

    tsom = TSOM2(X,latent_dim=2,resolution=10,SIGMA_MAX=2.0,SIGMA_MIN=0.2,TAU=50)
    tsom.fit(nb_epoch=50)

    comp=TSOM2_V(y=tsom.Y,winner1=tsom.k_star1,winner2=tsom.k_star2,
                 label1=labels_animal,label2=labels_feature)
    comp.draw_map()