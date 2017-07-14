class ObservationSpace(object):
    def __init__(self, axes, kse, show_f=True):
        self.axes = axes
        self.kse = kse
        self.x_style = 'k+'
        self.y_style = 'r.'

        self.show_f = show_f
        if show_f:
            self.kse.calcF(resolution=10)
            self.f_style = 'c-s'

    def update(self, epoch):
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])
        if self.axes.name == '3d':
            self.graph_y.set_3d_properties(Y[:, 2])

        if self.show_f:
            F = self.kse.history['f'][epoch, :, :]
            if self.axes.name == '3d':
                F = F.reshape(self.kse.resolution, self.kse.resolution, -1)
                self.axes.collections.remove(self.graph_f)
                self.graph_f = self.axes.plot_wireframe(F[:, :, 0], F[:, :, 1], F[:, :, 2], color=self.f_style[0], label="$F$")
            else:
                self.graph_f.set_xdata(F[:, 0])
                self.graph_f.set_ydata(F[:, 1])

    def init(self):
        Y = self.kse.history['y'][0, :, :]
        if self.show_f:
            F = self.kse.history['f'][0, :, :]

        if self.axes.name == '3d':
            self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.kse.X[:, 2], self.x_style, label="$X$")
            if self.show_f:
                F = F.reshape(self.kse.resolution, self.kse.resolution, -1)
                self.graph_f = self.axes.plot_wireframe(F[:, :, 0], F[:, :, 1], F[:, :, 2], color=self.f_style[0], label="$F$")
            self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], Y[:, 2], self.y_style, label="Y")
        else:
            self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.x_style, label="$X$")
            if self.show_f:
                self.graph_f, = self.axes.plot(F[:, 0], F[:, 1], self.f_style, label="$F$")
            self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], self.y_style, label="Y")

        self.axes.legend(loc='upper right')
