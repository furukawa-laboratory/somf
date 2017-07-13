class SequentialSpace(object):
    def __init__(self, axes, kse):
        self.axes = axes
        self.kse = kse

    def update(self, epoch):
        self.vline.set_xdata(x=epoch)

    def init(self):
        self.axes.axhline(y=0, linewidth=1, color='gray', linestyle='dashed')

        self.axes.plot(self.kse.history['gamma'], label='gamma', linewidth=1)
        self.axes.legend(loc='upper right')

        self.vline = self.axes.axvline(x=0, linewidth=1, color='k')
