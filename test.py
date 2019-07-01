import numpy as np
import matplotlib.pyplot as plt


def show(pic, color="gray", name="Picture", fontdict={'fontsize': 27}):

    plt.figure(figsize=(10,10))
    fig = plt.subplot(1,1,1)
    fig.imshow(pic, color)
    fig.set_title(name, fontdict)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()


d = np.load("depth.npy")

a = d[512][:]
show(d)
