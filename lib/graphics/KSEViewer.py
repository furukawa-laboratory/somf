import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
from lib.graphics.observation_space import ObservationSpace
from lib.graphics.sequential_space import SequentialSpace


class KSEViewer(object):
    def __init__(self, kse, rows, cols, figsize=None, skip=1):
        self.kse = kse
        self.rows = rows
        self.cols = cols
        if figsize is None:
            size = 3
            self.fig = plt.figure(figsize=(size*self.cols, size*self.rows))
        else:
            self.fig = plt.figure(figsize=figsize)
        self.skip = skip

        self.animation = None
        self.interval = 10
        self.nb_epoch = self.kse.history['z'].shape[0]

        self.spaces = []

    def add_observation_space(self, kse=None, row=1, col=1, **kwargs):
        index = (row-1) * self.cols + col
        axes = self.fig.add_subplot(self.rows, self.cols, index, **kwargs)
        if kse is None:
            self.spaces.append(ObservationSpace(axes, self.kse))
        else:
            self.spaces.append(ObservationSpace(axes, kse))


    def add_sequential_space(self, subject_name_list, kse=None, row=1, col=1, **kwargs):
        index = (row-1) * self.cols + col
        axes = self.fig.add_subplot(self.rows, self.cols, index, **kwargs)

        if kse is None:
            ss = SequentialSpace(axes, self.kse)
        else:
            ss = SequentialSpace(axes, kse)

        for subject_name in subject_name_list:
            ss.add_subject(subject_name)
        self.spaces.append(ss)

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
        for space in self.spaces:
            space.update(epoch)

    def _init(self):
        for i, space in enumerate(self.spaces):
            space.init()

    def save_gif(self, filename):
        self.animation.save(filename, writer='imagemagick', dpi=144)
