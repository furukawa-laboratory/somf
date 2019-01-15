import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
from sklearn.decomposition import PCA

def update_graph(epoch, X, Y_allepoch_mesh, labels, ax, title_text, nb_epoch):
    if epoch >= nb_epoch:
        epoch = nb_epoch - 1

    ax.cla()
    Y_mesh= Y_allepoch_mesh[epoch, :, :, :]
    """
    graph.set_data(data[:,:,0], data[:,:,1])
    graph.set_3d_properties(data[:,:,2])
    """
    ax.scatter(X[:,0],X[:,1],X[:,2],color='g')
    if labels is not None:
        for n in range(len(labels)):
            ax.text(X[n,0],X[n,1],X[n,2],labels[n])
    ax.plot_wireframe(Y_mesh[:, :, 0],
                      Y_mesh[:, :, 1],
                      Y_mesh[:, :, 2],
                      color='b')
    ax.set_title(title_text+', epoch={}'.format(epoch))


def anime_learning_process_3d(X, Y_allepoch, labels=None,
                              title_text='observation space',
                              repeat=True, save_gif=False):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,aspect='equal', projection='3d')
    ax_pos = ax.get_position()


    num_nodes = Y_allepoch.shape[1]
    ob_dim = Y_allepoch.shape[2]
    nb_epoch = Y_allepoch.shape[0]
    resolution = np.int(np.sqrt(num_nodes))


    if ob_dim > 3:
        pca = PCA(n_components=3)
        pca.fit(X)
        # W = np.diag(np.sqrt(pca.explained_variance_)) @ pca.components_
        W = pca.components_
        ev_ratio = pca.explained_variance_ratio_
        cumulative_propotion = ev_ratio.cumsum()[2]
        X = X @ W.T
        Y_allepoch = np.einsum("ekd,cd->ekc",Y_allepoch,W)
        ob_dim = 3
        fig.text(ax_pos.x0, ax_pos.y0,
                 'using PCA to visualize\ncumulative_propotion={:.3f}'.format(cumulative_propotion))

    Y_allepoch_mesh=Y_allepoch.reshape((nb_epoch, resolution, resolution, ob_dim))

    # ax.set_xlim(X[:,0].min(),X[:,0].max())
    # ax.set_ylim(X[:,1].min(),X[:,1].max())
    # ax.set_zlim(X[:,2].min(),X[:,2].max())
    Xmin = np.min(X, axis=0)
    Xmax = np.max(X, axis=0)
    Xmid = (Xmax + Xmin) * 0.5
    Xrange = np.max((Xmax - Xmin) * 0.5)
    ax.set_xlim(Xmid[0] - Xrange, Xmid[0] + Xrange)
    ax.set_ylim(Xmid[1] - Xrange, Xmid[1] + Xrange)
    ax.set_zlim(Xmid[2] - Xrange, Xmid[2] + Xrange)

    if repeat or save_gif:
        num_frames = nb_epoch
    else:
        num_frames = nb_epoch * 5
    ani = matplotlib.animation.FuncAnimation(fig, update_graph,
                                             fargs=(X,Y_allepoch_mesh,labels,ax,title_text,nb_epoch),
                                             interval=40, blit=False, repeat=repeat, frames=num_frames)

    if save_gif:
        ani.save(title_text+'.gif', writer='imagemagick', fps=10)
    else:
        plt.show()
