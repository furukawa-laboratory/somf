#人工データをplus型階層TSOM(TSOM*SOM)に適用したプログラムを追加する.
import numpy as np
from libs.datasets.artificial.kura_tsom import load_kura_tsom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#人工データの検証
X,z=load_kura_tsom(xsamples=10,ysamples=10,retz=True)
print(X.shape)
print(z.shape)

#人工データの描画
X_view=X.reshape((10*10,3))
Z_view=z.reshape((10*10,2))
fig=plt.figure()
ax=fig.add_subplot(1,2,1,projection="3d")
ax.scatter(X_view[:,0],X_view[:,1],X_view[:,2])
ax2=fig.add_subplot(1,2,2)
ax2.scatter(Z_view[:,0],Z_view[:,1])
plt.show()