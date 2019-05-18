#人工データをplus型階層TSOM(TSOM*SOM)に適用したプログラムを追加する.
import numpy as np
from libs.datasets.artificial.kura_tsom import load_kura_tsom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#人工データの検証
xsamples=20
ysamples=20
X,z=load_kura_tsom(xsamples=xsamples,ysamples=ysamples,retz=True)

#観測データを4分割
group1=X[0:10,0:10,:]
group2=X[0:10,10:20,:]
group3=X[10:20,0:10,:]
group4=X[10:20,10:20,:]


#人工データの描画
fig=plt.figure()
ax=fig.add_subplot(1,1,1,projection="3d")
#ax.scatter(X_view[:,0],X_view[:,1],X_view[:,2])
ax.scatter(group1[:,:,0].flatten(),group1[:,:,1].flatten(),group1[:,:,2].flatten(),color="red")
ax.scatter(group2[:,:,0].flatten(),group2[:,:,1].flatten(),group2[:,:,2].flatten(),color="blue")
ax.scatter(group3[:,:,0].flatten(),group3[:,:,1].flatten(),group3[:,:,2].flatten(),color="green")
ax.scatter(group4[:,:,0].flatten(),group4[:,:,1].flatten(),group4[:,:,2].flatten(),color="orange")
plt.show()

