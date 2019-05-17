import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


from libs.models.som import SOM
from libs.datasets.artificial import animal
from tests.SOMg.confirm_SOMg import SOMg #color_scaleをUmatrix_color_scale.txtに変更したconfirm_SOMgに変更
#from libs.visualization.som.somg import SOMg

if __name__ == '__main__':
    nb_epoch = 50
    resolution = 20
    sigma_max = 2.2
    sigma_min = 0.3
    tau = 50
    latent_dim = 2
    seed = 1

    title_text= "animal map"
    umat_resolution = 100 # U-matrix表示の解像度

    X, labels = animal.load_data()

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau)
    som.fit(nb_epoch=nb_epoch)

    #学習結果をresult.txtで用意した学習結果にすり替える
    Y=np.loadtxt("result.txt")#400*21
    som.Y=Y
    for i in range(nb_epoch):
        som.history["y"][i,:,:]=Y
    somg = SOMg(som)

    # 学習過程の表示
    fig, ax = plt.subplots()
    def calc_unit_distance(k1, k2):
        return np.linalg.norm(som.Y[k1] - som.Y[k2])
    somg.plot_umatrix(ax, calc_unit_distance)
    plt.show()

    # 学習過程の表示
    fig, ax = plt.subplots()
    def update(frame):
        if frame != 0:
            ax.cla()
        def calc_unit_distance_animation(k1, k2):
            return np.linalg.norm(som.history['y'][frame][k1] - som.history['y'][frame][k2])
        somg.plot_umatrix(ax, calc_unit_distance_animation)
        ax.set_title('epoch:{:03}'.format(frame))

    ani = animation.FuncAnimation(fig, update, frames=nb_epoch, interval=100)
    plt.show()
    # ani.save("somg.gif", writer='imagemagick')