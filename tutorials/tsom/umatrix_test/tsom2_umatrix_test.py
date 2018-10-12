from libs.visualization.tsom.tsom2_U_matrix import TSOM2_Umatrix
from libs.datasets.artificial.animal import load_data
import numpy as np
from libs.models.tsom import TSOM2


X,animal_label=load_data(retlabel_feature=False,retlabel_animal=True)

epoch_num=2


tsom2=TSOM2(X=X, latent_dim=2, resolution=20, SIGMA_MAX=2.0, SIGMA_MIN=0.2, TAU=50, init='random')

tsom2.fit(epoch_num)
z1=tsom2.history['z1'][epoch_num-1,:,:]
z2=tsom2.history['z2'][epoch_num-1,:,:]
y=tsom2.history['y'][epoch_num-1,:,:,:]

umat=TSOM2_Umatrix(z1=z1,z2=z2, x=X, sigma1=0.2,sigma2=0.2, resolution=10, labels1=animal_label,labels2=None, fig_size=[8,6], cmap_type='jet')
umat.draw_umatrix()
