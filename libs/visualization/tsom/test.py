import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(8*np.pi*t)

fig, ax = plt.subplots()
l, = ax.plot(t, s0, lw=2, color='red')
plt.subplots_adjust(right=0.8)

rax = plt.axes([0.8, 0.2, 0.1, 0.5], facecolor='lightgoldenrodyellow')
#label_box=['aa','bb','cc','dd']
label_box=np.arange(20)
radio = RadioButtons(rax, (label_box))

A=[s0,s1,s2]
def hzfunc(label):
    ydata = A[int(label)]
    l.set_ydata(ydata)
    plt.draw()

radio.on_clicked(hzfunc)
plt.show()