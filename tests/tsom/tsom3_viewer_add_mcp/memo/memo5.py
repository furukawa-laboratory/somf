import matplotlib.pyplot as plt

def motion(event):
    if event.dblclick == 1:
        plt.title("double click")

    elif event.button == 1:
        plt.title("left click")

    elif event.button == 3:
        plt.title("right click")

    plt.draw()

plt.figure()
plt.connect('button_press_event', motion)
plt.show()