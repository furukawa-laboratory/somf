import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

def update_graph(t,X,re_allY,ax,wframe,title):
    ax.cla()
    re_Y=re_allY[:,:,:,t]
    """
    graph.set_data(data[:,:,0], data[:,:,1])
    graph.set_3d_properties(data[:,:,2])
    """
    ax.scatter(X[:,0],X[:,1],X[:,2],color='g')
    wframe = ax.plot_wireframe(re_Y[:,:,0],re_Y[:,:,1],re_Y[:,:,2],color='b')
    title.set_text('SOM, time={}'.format(t))

def anime_reference_vector_3d(X,allY,resolution):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    D = allY.shape[1]
    T = allY.shape[2]
    re_allY=allY.reshape((resolution,resolution,D,T))
    ax.scatter(X[:,0],X[:,1],X[:,2], color='g')
    wframe = ax.plot_wireframe(re_allY[:,:,0,0],re_allY[:,:,1,0],re_allY[:,:,2,0],color='b')
    ax.set_xlim(X[:,0].min(),X[:,0].max())
    ax.set_ylim(X[:,1].min(),X[:,1].max())
    ax.set_zlim(X[:,2].min(),X[:,2].max())

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, fargs=(X,re_allY,ax,wframe,title),
                                   interval=40, blit=False,frames=T)

    plt.show()
