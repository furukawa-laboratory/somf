

import numpy as np
from tsom3_viewer_mcp import TSOM3_Viewer

tsom3_Y = np.load('tsom3_Y.npy')
tsom3_k1 = np.load('tsom3_k1.npy')
tsom3_k2 = np.load('tsom3_k2.npy')
tsom3_k3 = np.load('tsom3_k3.npy')

tsom3_V = TSOM3_Viewer(tsom3_Y, tsom3_k1, tsom3_k2, tsom3_k3)
tsom3_V.draw_map()
