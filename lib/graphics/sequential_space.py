class SequentialSpace(object):
    def __init__(self, axes, kse):
        self.axes = axes
        self.kse = kse
        self.subject_name_list = []

    def add_subject(self, subject_name):
        self.subject_name_list.append(subject_name)

    def update(self, epoch):
        self.vline.set_xdata(x=epoch)

    def init(self):
        self.axes.axhline(y=0, linewidth=1, color='gray', linestyle='dashed')

        for subject_name in self.subject_name_list:
            self.axes.plot(self.kse.history[subject_name], label=subject_name, linewidth=1)
            self.axes.legend(loc='upper right')

        self.vline = self.axes.axvline(x=0, linewidth=1, color='k')
