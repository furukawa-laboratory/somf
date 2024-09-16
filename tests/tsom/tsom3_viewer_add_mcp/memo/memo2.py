import matplotlib.pyplot as plt

def motion(event):
    if (event.xdata is None) or (event.ydata is None):
        return

    x = event.xdata
    y = event.ydata

    ln_v.set_xdata(x)
    ln_h.set_ydata(y)
    plt.draw()

plt.figure()
ln_v = plt.axvline(0)
ln_h = plt.axhline(0)

plt.connect('motion_notify_event', motion)
plt.show()