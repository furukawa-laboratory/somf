class ObservationSpace(object):
    def __init__(self, axes, kse):
        self.axes = axes
        self.kse = kse
        self.x_kws = 'k+'
        self.y_kws = 'r.'

    def update(self, epoch):
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])

    def init(self):
        self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.x_kws, label="$X$")
        Y = self.kse.history['y'][0, :, :]
        self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], self.y_kws, label="Y")
        self.axes.legend(loc='upper right')
