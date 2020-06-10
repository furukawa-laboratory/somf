import matplotlib.pyplot as plt
from libs.datasets.artificial.kura_tsom import load_kura_tsom
from libs.models.tsom import TSOM2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from libs.models.tsom_cross_som import TSOM_cross_SOM
import numpy as np

#データの読み込み
I=30
J=10
seed = 2
np.random.seed(seed)
nb_epoch = 50
n_class = 6
n_sample_1 = I
n_sample_2 = J
Dim = 3
parent_latent_dim = 2
child_latent_dim = [1, 1]
parent_resolution = 8
child_resolution = [50, 50]
# pCluster_num = parent_resolution ** parent_latent_dim
# cCluster_num = child_resolution ** child_latent_dim
parent_sigma_max = 2.0
parent_sigma_min = 0.3
child_sigma_max = 2.0
child_sigma_min = 0.15
parent_tau = nb_epoch
child_tau = nb_epoch
interval = 500
data = load_kura_tsom(xsamples=I,ysamples=J)
print(data.shape)
Datasets = np.zeros((n_class, n_sample_1, n_sample_2, Dim))
for n in range(n_class):
    Datasets[n, :, :, :] = data

pZ = None
cZ = None

# nodes1_kx=10
# nodes1_ky=1#kuraの場合,潜在空間は1次元
# nodes1_num=nodes1_kx*nodes1_ky
# nodes2_kx=5
# nodes2_ky=1#kuraの場合,潜在空間は1次元
# nodes2_num=nodes2_kx*nodes2_ky
# mode1_samples=X.shape[0]
# mode2_samples=X.shape[1]
# observed_dim=X.shape[2]
# tau1=50
# tau2=50
# sigma1_min=0.1
# sigma1_zero=1.2
# sigma2_min=0.1
# sigma2_zero=1.2


tsom2=TSOM_cross_SOM(Datasets, parent_latent_dim, child_latent_dim, parent_resolution, child_resolution,
                 parent_sigma_max, child_sigma_max, parent_sigma_min, child_sigma_min,
                 parent_tau, child_tau, pZ, cZ)
tsom2.fit(nb_epoch=5)
#観測空間の描画

fig = plt.figure()
ax = Axes3D(fig)
n = 0
def plot(i):
    ax.cla()
    ax.scatter(data[:,:, 0], data[:,:, 1], data[:,:, 2])
    ax.plot_wireframe(tsom2.history['cY'][i,n,:, :, 0], tsom2.history['cY'][i,n,:, :, 1], tsom2.history['cY'][i,n,:, :, 2])
    plt.title(' t=' + str(i))

ani = animation.FuncAnimation(fig, plot, frames=250,interval=100)
plt.show()

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