import matplotlib.pyplot as plt

def motion(event):
    if (event.xdata is None) or (event.ydata is None):
        return

    x = event.xdata
    y = event.ydata

    ln.set_data(x,y)
    plt.draw()

plt.figure()
ln, = plt.plot([],[],'x')

plt.connect('motion_notify_event', motion)
plt.show()