import matplotlib.pyplot as plt
from libs.datasets.artificial.kura_tsom import load_kura_tsom
from libs.models.tsom import TSOM2
from libs.models.tsom3 import TSOM3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V
from libs.visualization.tsom.tsom3_viewer import TSOM3_Viewer as TSOM3_V

#データの読み込み
I=30
J=10
X=load_kura_tsom(I,J)
print(X.shape)

nodes1_kx=10
nodes1_ky=1#kuraの場合,潜在空間は1次元
nodes1_num=nodes1_kx*nodes1_ky
nodes2_kx=5
nodes2_ky=1#kuraの場合,潜在空間は1次元
nodes2_num=nodes2_kx*nodes2_ky
mode1_samples=X.shape[0]
mode2_samples=X.shape[1]
observed_dim=X.shape[2]
tau1=50
tau2=50
sigma1_min=0.1
sigma1_zero=1.2
sigma2_min=0.1
sigma2_zero=1.2


# tsom2=TSOM2(X,latent_dim=(1,1),resolution=(10,15),SIGMA_MAX=(sigma1_zero,sigma2_zero),
#                   SIGMA_MIN=sigma1_min, TAU=(tau1,tau2))
# tsom2.fit(nb_epoch=250)

tsom3=TSOM3(X,latent_dim=2,resolution=5,SIGMA_MAX=2.0,
                  SIGMA_MIN=0.2, TAU=(50, 50 ,50))
tsom3.fit(nb_epoch=250)

#観測空間の描画
# comp=TSOM2_V(y=tsom3.Y,winner1=tsom3.k1_star,winner2=tsom3.k2_star)
comp=TSOM3_V(y=tsom3.Y,winner1=tsom3.k1_star,winner2=tsom3.k2_star,winner3=tsom3.k3_star)
comp.draw_map()

# 結果の描画
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# ax.scatter(tsom3.Z2[:, 0], tsom3.Z2[:, 1])
# for i in range(X.shape[1]):
#     ax.text(tsom3.Z2[i, 0], tsom3.Z2[i, 1], beverage_label[i])

# ax2 = fig.add_subplot(1, 2, 2)
# ax2.scatter(tsom3.Z3[:, 0], tsom3.Z3[:, 1])
# for i in range(X.shape[2]):
#     ax2.text(tsom3.Z3[i, 0], tsom3.Z3[i, 1], situation_label[i])
# plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# def plot(i):
#     ax.cla()
#     ax.scatter(X[:,:, 0], X[:,:, 1], X[:,:, 2])
#     ax.plot_wireframe(tsom3.history['y'][i,:, :, 0], tsom3.history['y'][i,:, :, 1], tsom3.history['y'][i,:, :, 2])
#     plt.title(' t=' + str(i))
#
# ani = animation.FuncAnimation(fig, plot, frames=250,interval=100)
# plt.show()

#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:,:, 0], X[:,:, 1], X[:,:, 2])
# ax.plot_wireframe(tsom2.history['y'][249,:, :, 0], tsom2.history['y'][249,:, :, 1], tsom2.history['y'][249,:, :, 2])
#
# #ani = animation.FuncAnimation(fig, plot, frames=250,interval=100)
# plt.show()





#
# #X=X.reshape(mode1_samples,mode2_samples)
# #潜在空間の描画
# #Umatrix表示
# #modeごとで選べるようにしたいね
# #X=X.reshape((X.shape[0],X.shape[1]))
# # umatrix1=TSOM2_Umatrix( z1=tsom2.Z1,z2=tsom2.Z2, x=X, sigma1=sigma1_min,sigma2=sigma2_min, resolution=20, labels1=category_label,labels2=features_label)
# # umatrix1.draw_umatrix()
#
# # #C
#
#
# # fig = plt.figure()
# # plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1],c=category_label)
# # #plt.scatter(tsom_kl.Zv[:,0],tsom_kl.Zv[:,1])
# # for i in np.arange(X.shape[0]):
# #     plt.annotate(category_label[i], (tsom_kl.Zd[i, 0], tsom_kl.Zd[i, 1]))
# # # for j in np.arange(X.shape[1]):
# # #     plt.text( tsom_kl.Zv[j, 0], tsom_kl.Zv[j, 1],features_label[j])
# #
# # plt.show()
#
#
#
#
# #
# # fig = plt.figure()
# # plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1],c=category_label)
# # #plt.scatter(tsom_kl.Zv[:,0],tsom_kl.Zv[:,1])
# # for i in np.arange(X.shape[0]):
# #     plt.annotate(category_label[i], (tsom_kl.Zd[i, 0], tsom_kl.Zd[i, 1]))
# # # for j in np.arange(X.shape[1]):
# # #     plt.text( tsom_kl.Zv[j, 0], tsom_kl.Zv[j, 1],features_label[j])
# # plt.show()
# # fig1 = plt.figure()
# # #plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1])
# # plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1],c=category_label)
# # epsilon = 0.04 * (tsom_kl.Zd.max() - tsom_kl.Zd.min())
# # for i in range(X.shape[0]):
# #     count = 0
# #     for j in range(i):
# #         if np.allclose(tsom_kl.Zd[j, :], tsom_kl.Zd[i, :]):
# #             count += 1
# #     plt.text(tsom_kl.Zd[i, 0], tsom_kl.Zd[i, 1] + epsilon * count, category_label[i], color='k')
# # plt.show()
# #
# #
# # fig2 = plt.figure()
# # plt.scatter(tsom_kl.Zv[:,0],tsom_kl.Zv[:,1])
# # epsilon = 0.04 * (tsom_kl.Zv.max() - tsom_kl.Zv.min())
# # for i in range(X.shape[1]):
# #     count = 0
# #     for j in range(i):
# #         if np.allclose(tsom_kl.Zv[j, :], tsom_kl.Zv[i, :]):
# #             count += 1
# #     plt.text(tsom_kl.Zv[i, 0], tsom_kl.Zv[i, 1] + epsilon * count, features_label[i], color='k')
# # plt.show()