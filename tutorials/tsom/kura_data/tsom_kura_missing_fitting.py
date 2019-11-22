import matplotlib.pyplot as plt
from libs.datasets.artificial.kura_tsom import load_kura_tsom
from libs.models.tsom import TSOM2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation



X,Gamma=load_kura_tsom(xsamples=10,ysamples=20,missing_rate=0.7)


tau1=50
tau2=50
sigma1_min=0.1
sigma1_zero=1.2
sigma2_min=0.1
sigma2_zero=1.2


tsom2=TSOM2(X,latent_dim=(1,1),resolution=(30,20),SIGMA_MAX=(sigma1_zero,sigma2_zero),
                  SIGMA_MIN=sigma1_min, TAU=(tau1,tau2),model = 'indirect')
tsom2.fit(nb_epoch=250)
#観測空間の描画

fig = plt.figure()
ax = Axes3D(fig)
def plot(i):
    ax.cla()
    ax.scatter(X[:,:, 0], X[:,:, 1], X[:,:, 2])
    ax.plot_wireframe(tsom2.history['y'][i,:, :, 0], tsom2.history['y'][i,:, :, 1], tsom2.history['y'][i,:, :, 2])
    plt.title(' t=' + str(i))

ani = animation.FuncAnimation(fig, plot, frames=250,interval=100)
plt.show()