class ObservationSpace(object):
    def __init__(self, fig, kse):
        self.fig = fig
        self.kse = kse
        self.axes = None

    def update(self, epoch):
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])

    def init(self, *args, **kwargs):
        self.axes = self.fig.add_subplot(*args, **kwargs)

        self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], 'o', label="$X$")
        Y = self.kse.history['y'][0, :, :]
        self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], 'x', label="Y")
        self.axes.legend(loc='upper right')