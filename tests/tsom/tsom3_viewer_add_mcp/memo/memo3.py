import matplotlib.pyplot as plt

def motion(event):
    x = event.xdata
    y = event.ydata

    if event.inaxes == ax1:
        ln_1.set_data(x,y)
    if event.inaxes == ax2:
        ln_2.set_data(x,y)
    plt.draw()

plt.figure()
ax1 = plt.subplot(1,2,1)
ln_1, = plt.plot([],[],'o')

ax2 = plt.subplot(1,2,2)
ln_2, = plt.plot([],[],'x')

plt.connect('motion_notify_event', motion)
plt.show()