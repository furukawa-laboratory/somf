#人工データをplus型階層TSOM(TSOM*SOM)に適用したプログラムを追加する.
import numpy as np
from libs.datasets.artificial.kura_tsom import load_kura_tsom
import matplotlib.pyplot as plt
from libs.models.TSOM_plus_SOM import TSOM_plus_SOM
from mpl_toolkits.mplot3d import Axes3D
from libs.visualization.som.Grad_norm import Grad_Norm

#人工データの検証
xsamples=20
ysamples=20
X,z=load_kura_tsom(xsamples=xsamples,ysamples=ysamples,retz=True)
#観測データを4分割
group1=X[0:10,0:10,:]
group1=group1.reshape((int(xsamples*ysamples/4),3))
group2=X[0:10,10:20,:]
group2=group2.reshape((int(xsamples*ysamples/4),3))
group3=X[10:20,0:10,:]
group3=group3.reshape((int(xsamples*ysamples/4),3))
group4=X[10:20,10:20,:]
group4=group4.reshape((int(xsamples*ysamples/4),3))


#人工データの描画
fig=plt.figure()
ax=fig.add_subplot(1,1,1,projection="3d")
#ax.scatter(X_view[:,0],X_view[:,1],X_view[:,2])
ax.scatter(group1[:,0],group1[:,1],group1[:,2],color="red")
ax.scatter(group2[:,0],group2[:,1],group2[:,2],color="blue")
ax.scatter(group3[:,0],group3[:,1],group3[:,2],color="green")
ax.scatter(group4[:,0],group4[:,1],group4[:,2],color="orange")
#plt.show()

#グループラベルの作成
group1_label=np.arange(0,100)
group2_label=np.arange(100,200)
group3_label=np.arange(200,300)
group4_label=np.arange(300,400)
group_label=(group1_label,group2_label,group3_label,group4_label)


input_data=np.concatenate((group1,group2,group3,group4),axis=0)
args=((2,2),(10,10),(1.0,1.0),(0.1,0.1),(50,50))

#+型階層TSOMのclass読み込み
tsom_plus_som=TSOM_plus_SOM(input_data,"random",group_label,(2,2),(10,10),(1.0,1.0),(0.1,0.1),(50,50))

tsom_plus_som.fit_1st_TSOM(tsom_epoch_num=250)
tsom_plus_som.fit_KDE(kernel_width=1.0)
tsom_plus_som.fit_2nd_SOM(som_epoch_num=250,init="random")#2ndSOMの学習

#grad_normで可視化
som_umatrix = Grad_Norm(X=tsom_plus_som.som.X,Z=tsom_plus_som.som.Z,sigma=0.1,labels=None,title_text="team_map",resolution=20)
som_umatrix.draw_umatrix()