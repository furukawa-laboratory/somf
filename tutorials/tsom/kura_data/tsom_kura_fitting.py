import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from visualization.tsom.animation_learning_process_3d import anime_learning_process_3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


#データの読み込み
# I=30
# J=30
# X=load_kura_tsom(I,J)
# print(X.shape)
#print(features_label)
# #X,sushilabel=load_animal()
#X,animal_label,feature_label=load_animal()
# X,sushilabel=tsom_load_sushi()
#print(X.shape)

nodes1_kx=10
nodes1_ky=10
nodes1_num=nodes1_kx*nodes1_ky
nodes2_kx=10
nodes2_ky=10
nodes2_num=nodes2_kx*nodes2_ky
mode1_samples=X.shape[0]
mode2_samples=X.shape[1]
observed_dim=X.shape[2]
tau1=100
tau2=100
sigma1_min=0.1
sigma1_zero=1.2
sigma2_min=0.1
sigma2_zero=1.2

epoch_num=500
#tsom2=TSOM2.TSOM2(X, mode1_nodes=[nodes1_kx,nodes1_ky],mode2_nodes=[nodes2_kx,nodes2_ky],SIGMA_MAX=[sigma1_zero, sigma2_zero] ,SIGMA_MIN=[sigma1_min, sigma2_min], TAU=[tau1,tau2],epoch_num=epoch_num)
tsom_kl=TSOMKL.TSOMKL(X,X2, mode1_nodes=[nodes1_kx,nodes1_ky],mode2_nodes=[nodes2_kx,nodes2_ky],SIGMA_MAX=[sigma1_zero, sigma2_zero] ,SIGMA_MIN=[sigma1_min, sigma2_min], TAU=[tau1,tau2],epoch_num=epoch_num)
#Y_allepoch=np.zeros((epoch_num,nodes1_num,nodes2_num,observed_dim))

for epoch in tqdm(range(epoch_num)):
    tsom_kl.learning(epoch)
    #Y_allepoch[epoch,:,:,:]=Y

#ラベルが重ならないようにする必要がある
# fig = plt.figure()
# plt.scatter(tsom2.Z1[:, 0], tsom2.Z1[:, 1], c=category_label)
# for i in np.arange(X.shape[0]):
#     plt.annotate(category_label[i], (tsom2.Z1[i, 0], tsom2.Z1[i, 1]))
# plt.show()
#
# fig = plt.figure()
# plt.scatter(tsom2.Z2[:, 0], tsom2.Z2[:, 1])
# for i in np.arange(X.shape[1]):
#     plt.annotate(features_label[i], (tsom2.Z2[i, 0], tsom2.Z2[i, 1]))
# plt.show()

print(tsom_kl.zd)
#print(tsom_kl.Zv)

# # #観測空間の描画
# fig = plt.figure()
# ax = Axes3D(fig)
# def plot(i):
#     ax.cla()
#     ax.scatter(X[:,:, 0], X[:,:, 1], X[:,:, 2])
#     ax.plot_wireframe(Y_allepoch[i,:, :, 0], Y_allepoch[i,:, :, 1], Y_allepoch[i,:, :, 2])
#     plt.title(' t=' + str(i))
# ani = animation.FuncAnimation(fig, plot, frames=epoch_num,interval=100)
# #learning_process_tsom(X,Y_allepoch)
# plt.show()

#X=X.reshape(mode1_samples,mode2_samples)
#潜在空間の描画
#Umatrix表示
#modeごとで選べるようにしたいね
#X=X.reshape((X.shape[0],X.shape[1]))
# umatrix1=TSOM2_Umatrix( z1=tsom2.Z1,z2=tsom2.Z2, x=X, sigma1=sigma1_min,sigma2=sigma2_min, resolution=20, labels1=category_label,labels2=features_label)
# umatrix1.draw_umatrix()

# #C


# fig = plt.figure()
# plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1],c=category_label)
# #plt.scatter(tsom_kl.Zv[:,0],tsom_kl.Zv[:,1])
# for i in np.arange(X.shape[0]):
#     plt.annotate(category_label[i], (tsom_kl.Zd[i, 0], tsom_kl.Zd[i, 1]))
# # for j in np.arange(X.shape[1]):
# #     plt.text( tsom_kl.Zv[j, 0], tsom_kl.Zv[j, 1],features_label[j])
#
# plt.show()




#
# fig = plt.figure()
# plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1],c=category_label)
# #plt.scatter(tsom_kl.Zv[:,0],tsom_kl.Zv[:,1])
# for i in np.arange(X.shape[0]):
#     plt.annotate(category_label[i], (tsom_kl.Zd[i, 0], tsom_kl.Zd[i, 1]))
# # for j in np.arange(X.shape[1]):
# #     plt.text( tsom_kl.Zv[j, 0], tsom_kl.Zv[j, 1],features_label[j])
# plt.show()
# fig1 = plt.figure()
# #plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1])
# plt.scatter(tsom_kl.Zd[:,0],tsom_kl.Zd[:,1],c=category_label)
# epsilon = 0.04 * (tsom_kl.Zd.max() - tsom_kl.Zd.min())
# for i in range(X.shape[0]):
#     count = 0
#     for j in range(i):
#         if np.allclose(tsom_kl.Zd[j, :], tsom_kl.Zd[i, :]):
#             count += 1
#     plt.text(tsom_kl.Zd[i, 0], tsom_kl.Zd[i, 1] + epsilon * count, category_label[i], color='k')
# plt.show()
#
#
# fig2 = plt.figure()
# plt.scatter(tsom_kl.Zv[:,0],tsom_kl.Zv[:,1])
# epsilon = 0.04 * (tsom_kl.Zv.max() - tsom_kl.Zv.min())
# for i in range(X.shape[1]):
#     count = 0
#     for j in range(i):
#         if np.allclose(tsom_kl.Zv[j, :], tsom_kl.Zv[i, :]):
#             count += 1
#     plt.text(tsom_kl.Zv[i, 0], tsom_kl.Zv[i, 1] + epsilon * count, features_label[i], color='k')
# plt.show()