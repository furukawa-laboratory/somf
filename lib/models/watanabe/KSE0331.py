import numpy as np


class KSE(object):
    def __init__(self):
        self.x = 2
        self.z = np.linspace(-1, 1)
        self.z += 0.0000001
