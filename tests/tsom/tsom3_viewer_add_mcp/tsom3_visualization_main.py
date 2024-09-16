

import numpy as np
from tsom3_viewer import TSOM3_Viewer

tsom3_Y = np.load('tsom3_Y.npy')
tsom3_k1 = np.load('tsom3_k1.npy')
tsom3_k2 = np.load('tsom3_k2.npy')
tsom3_k3 = np.load('tsom3_k3.npy')

label1 = np.arange(1,11)
label2 = np.loadtxt("beverage_label.txt", dtype="str")
label3 = np.loadtxt("situation_label.txt", dtype="str")

tsom3_V = TSOM3_Viewer(tsom3_Y, tsom3_k1, tsom3_k2, tsom3_k3, label2=label2, label3=label3, view1_title='user', view2_title='beverage', view3_title='situation')
tsom3_V.draw_map()

