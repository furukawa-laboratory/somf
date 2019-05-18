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
#人工データの描画
X_view=X.reshape((xsamples*ysamples,3))
group1=X_view[0:100,:]
group2=X_view[100:200,:]
group3=X_view[200:300,:]
group4=X_view[300:400,:]


fig=plt.figure()
ax=fig.add_subplot(1,1,1,projection="3d")
#ax.scatter(X_view[:,0],X_view[:,1],X_view[:,2])
ax.scatter(group1[:,0],group1[:,1],group1[:,2],color="red")
ax.scatter(group2[:,0],group2[:,1],group2[:,2],color="blue")
ax.scatter(group3[:,0],group3[:,1],group3[:,2],color="green")
ax.scatter(group4[:,0],group4[:,1],group4[:,2],color="orange")
plt.show()