class ObservationSpace(object):
    def __init__(self, axes, kse):
        self.axes = axes
        self.kse = kse
        self.x_style = 'k+'
        self.y_style = 'r.'

    def update(self, epoch):
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])
        if self.axes.name == '3d':
            self.graph_y.set_3d_properties(Y[:, 2])

    def init(self):
        Y = self.kse.history['y'][0, :, :]

        if self.axes.name == '3d':
            self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.kse.X[:, 2], self.x_style, label="$X$")
            self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], Y[:, 2], self.y_style, label="Y")
        else:
            self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.x_style, label="$X$")
            self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], self.y_style, label="Y")

        self.axes.legend(loc='upper right')
