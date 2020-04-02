

import numpy as np
from tsom3_viewer_mcp import TSOM3_Viewer
from tsom2_viewer import TSOM2_Viewer

tsom3_Y = np.load('tsom3_Y.npy')
tsom3_k1 = np.load('tsom3_k1.npy')
tsom3_k2 = np.load('tsom3_k2.npy')
tsom3_k3 = np.load('tsom3_k3.npy')

# tsom2_Y = np.load('tsom2_Y.npy')
# tsom2_k1 = np.load('tsom2_k1.npy')
# tsom2_k2 = np.load('tsom2_k2.npy')

label1 = np.arange(1,11)
label2 = np.arange(1,11)
label3 = np.arange(1,4)

tsom3_V = TSOM3_Viewer(tsom3_Y, tsom3_k1, tsom3_k2, tsom3_k3, view1_title='leader', view2_title='follower', view3_title='action')
tsom3_V.draw_map()
#
# tsom2_V = TSOM2_Viewer(tsom2_Y, tsom2_k1, tsom2_k2)
# tsom2_V.draw_map()
