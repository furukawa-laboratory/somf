import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

def draw(X,history):

def update_graph():
    ax_observed.cla()
    re_Y=re_allY[:,:,:,t]
    if X != None:
        ax_observed.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=numbers)
    ax_observed.plot_wireframe(re_Y[:, :, 0], re_Y[:, :, 1], re_Y[:, :, 2])
    #潜在空間の表示
    ax_latent.cla()
    Z = allZ[:,:,t]
    #Zplot.set_data(Z[:,0],Z[:,1])
    ax_latent.set_xlim(1.1*allZ[:,0,:].min(),1.1*allZ[:,0,:].max())
    ax_latent.set_ylim(1.1*allZ[:,1,:].min(),1.1*allZ[:,1,:].max())
    ax_latent.scatter(Z[:,0],Z[:,1],marker="o",c=numbers)
    title.set_text('time={}'.format(t))
