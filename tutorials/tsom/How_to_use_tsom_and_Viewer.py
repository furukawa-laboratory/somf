from libs.models.tsom import TSOM2
from libs.datasets.real.beverage import load_data
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V


if __name__ == '__main__':
    objects=load_data()
    print(len(objects))
    X=objects[0]
    tsom = TSOM2(X,latent_dim=2,resolution=10,SIGMA_MAX=2.0,SIGMA_MIN=0.2,TAU=50)
    tsom.fit(nb_epoch=10)
    print(tsom.X.shape,tsom.Y.shape)
    comp=TSOM2_V(y=tsom.Y,winner1=tsom.k_star1,winner2=tsom.k_star2)
    comp.draw_map()