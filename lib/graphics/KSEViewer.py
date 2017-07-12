import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from lib.graphics.observation_space import ObservationSpace


class KSEViewer(object):
    def __init__(self, kse):
        self.kse = kse
        self.fig = plt.figure()

        self.animation = None
        self.interval = 10
        self.skip = 1
        self.nb_epoch = self.kse.history['z'].shape[0]

        self.spaces = []

    def add_observation_space(self):
        self.spaces.append(ObservationSpace(self.fig, self.kse))

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
        for space in self.spaces:
            space.init(111, aspect='equal')

    def save_gif(self, filename):
        self.animation.save(filename, writer='imagemagick', dpi=144)
