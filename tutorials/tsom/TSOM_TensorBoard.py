# coding: utf-8
import sys
import pprint
pprint.pprint(sys.path)
sys.path.append( '/Users/hatanohajime/Desktop/PycharmProjects/flib')

from tsom_tf import TSOM2
from libs.datasets.artificial.animal import load_data
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V

X, labels_animal, labels_feature = load_data(retlabel_animal=True, retlabel_feature=True)

X = X.reshape((X.shape[0], X.shape[1], 1))

tsom = TSOM2(N1=X.shape[0], N2=X.shape[1], observed_dim=1, latent_dim=2, epochs=50, resolution=10, SIGMA_MAX=2.0, SIGMA_MIN=0.2, TAU=50, init='random')
tsom.predict(X)

print('Drawing maps')
comp = TSOM2_V(y=tsom.historyY[-1], winner1=tsom.bmu1[-1], winner2=tsom.bmu2[-1],
             label1=labels_animal, label2=labels_feature)
comp.draw_map()

