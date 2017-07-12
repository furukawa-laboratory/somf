import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation


class KSEViewer(object):
    def __init__(self, kse):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111, aspect='equal')
        self.interval = 10
        self.kse = kse
        self.animation = None
        self.skip = 1
        self.nb_epoch = self.kse.history['z'].shape[0]

    def draw(self):
        frames = self.nb_epoch // self.skip

        self.animation = matplotlib.animation.FuncAnimation(self.fig,
                                                            self._update,
                                                            frames=frames,
                                                            init_func=self._init,
                                                            interval=self.interval,
                                                            repeat=False)
        plt.show()

    def _update(self, i):
        epoch = i * self.skip
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])

    def _init(self):
        self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], 'o', label="$X$")
        Y = self.kse.history['y'][0, :, :]
        self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], 'x', label="Y")
        self.axes.legend(loc='upper right')

    def save_gif(self, filename):
        self.animation.save(filename, writer='imagemagick', dpi=144)
