import numpy as np


#
def load_kura_tsom(xsamples,ysamples,retz=False):
    x=np.linspace(-1,1,xsamples)
    y=np.linspace(-1,1,ysamples)

    xx,yy=np.meshgrid(y,x)
    z=xx**2-yy**2

    x=np.concatenate((xx[:,:,np.newaxis],yy[:,:,np.newaxis],z[:,:,np.newaxis]),axis=2)


    if retz:
        return x, xx, yy
    else:
        return x




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(X[:,:,0],X[:,:,1],X[:,:,2])
    plt.show()