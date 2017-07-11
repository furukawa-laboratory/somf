import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

class KSEViewer(object):

    def __init__(self, X, history):
        self.X = X
        self.history = history
        self.axlist = []

    def draw(self,skip=1):
        #潜在空間，観測空間
        matplotlib.animation.FuncAnimation(fig, self._update(),
                                           interval=interval, blit=False,frames=T)
    def _update(self,t):
        for ax in self.axlist:
            ax.update(t)