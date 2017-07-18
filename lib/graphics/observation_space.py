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

        if self.axes.name == '3d':
            self.view_dim = 3
        else:
            self.view_dim = 2

        self.latent_dim = self.kse.L

        if (self.view_dim, self.latent_dim) == (3, 2):
            self.init = self.init_3dview_2dmesh
            self.update = self.update_3dview_2dmesh
        elif (self.view_dim, self.latent_dim) == (3, 1):
            self.init = self.init_3dview_1dmesh
            self.update = self.update_3dview_1dmesh
        elif (self.view_dim, self.latent_dim) == (2, 1):
            self.init = self.init_2dview_1dmesh
            self.update = self.update_2dview_1dmesh
        else:
            raise ValueError('invalid dim view:{0}, latent{1}'.format(self.view_dim, self.latent_dim))

    def update_3dview_2dmesh(self, epoch):
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])
        self.graph_y.set_3d_properties(Y[:, 2])
        if self.show_f:
            F = self.kse.history['f'][epoch, :, :]
            F = F.reshape(self.kse.resolution, self.kse.resolution, -1)
            self.axes.collections.remove(self.graph_f)
            self.graph_f = self.axes.plot_wireframe(F[:, :, 0], F[:, :, 1], F[:, :, 2], color=self.f_style[0], label="$F$")

    def update_3dview_1dmesh(self, epoch):
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])
        self.graph_y.set_3d_properties(Y[:, 2])
        if self.show_f:
            F = self.kse.history['f'][epoch, :, :]
            self.graph_f.set_xdata(F[:, 0])
            self.graph_f.set_ydata(F[:, 1])
            self.graph_f.set_3d_properties(F[:, 2])

    def update_2dview_1dmesh(self, epoch):
        Y = self.kse.history['y'][epoch, :, :]
        self.graph_y.set_xdata(Y[:, 0])
        self.graph_y.set_ydata(Y[:, 1])
        if self.show_f:
            F = self.kse.history['f'][epoch, :, :]
            self.graph_f.set_xdata(F[:, 0])
            self.graph_f.set_ydata(F[:, 1])

    def init_3dview_2dmesh(self):
        self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.kse.X[:, 2], self.x_style, label="$X$")
        if self.show_f:
            F = self.kse.history['f'][0, :, :]
            F = F.reshape(self.kse.resolution, self.kse.resolution, -1)
            self.graph_f = self.axes.plot_wireframe(F[:, :, 0], F[:, :, 1], F[:, :, 2], color=self.f_style[0], label="$F$")
        Y = self.kse.history['y'][0, :, :]
        self.graph_y, = self.axes.plot(Y[:, 0], Y[:, 1], Y[:, 2], self.y_style, label="Y")
        self.axes.legend(loc='upper right')

    def init_3dview_1dmesh(self):
        self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.kse.X[:, 2], self.x_style, label="$X$")
        if self.show_f:
            F = self.kse.history['f'][0, :, :]
            self.graph_f = self.axes.plot(F[:, 0], F[:, 1], F[:, 2], self.f_style, label="$F$")[0]
        Y = self.kse.history['y'][0, :, :]
        self.graph_y = self.axes.plot(Y[:, 0], Y[:, 1], Y[:, 2], self.y_style, label="Y")[0]
        self.axes.legend(loc='upper right')

    def init_2dview_1dmesh(self):
        self.axes.plot(self.kse.X[:, 0], self.kse.X[:, 1], self.x_style, label="$X$")
        if self.show_f:
            F = self.kse.history['f'][0, :, :]
            self.graph_f = self.axes.plot(F[:, 0], F[:, 1], self.f_style, label="$F$")[0]
        Y = self.kse.history['y'][0, :, :]
        self.graph_y = self.axes.plot(Y[:, 0], Y[:, 1], self.y_style, label="Y")[0]
        self.axes.legend(loc='upper right')
