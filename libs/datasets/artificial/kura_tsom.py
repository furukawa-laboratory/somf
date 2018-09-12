import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#
def load_kura_tsom(xsamples,ysamples):
    x=np.linspace(-1,1,xsamples)
    y=np.linspace(-1,1,ysamples)

    xx,yy=np.meshgrid(y,x)
    z=xx**2-yy**2

    X=np.concatenate((xx[:,:,np.newaxis],yy[:,:,np.newaxis],z[:,:,np.newaxis]),axis=2)


    return X





    # fig=plt.figure()
    # ax=Axes3D(fig)
    # ax.scatter(X[:,:,0],X[:,:,1],X[:,:,2])
    # plt.show()