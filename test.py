import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def normalize_normal_map(N):
    """
    N is a unnormalized normal map of shape H_W_3. Normalize N across the third dimension.
    :param N:
    :return:
    """
    H, W, C = N.shape
    N = np.reshape(N, (-1, C))
    N = normalize(N, axis=1)
    N = np.reshape(N, (H, W, C))
    return N


def show(pic, color="gray", name="Picture", fontdict={'fontsize': 27}):

    plt.figure(figsize=(10,10))
    fig = plt.subplot(1,1,1)
    fig.imshow(pic, color)
    fig.set_title(name, fontdict)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()


d = np.load("depth.npy")
n = normalize_normal_map(np.load("normal.npy"))

center = d.shape[0]//2
c = center

clip_size = 50
l = clip_size

a = d[c-l:c+l+1, c-l:c+l+1]
b = n[c-l:c+l+1, c-l:c+l+1]
b[:,:,0] /= b[:,:,2]
b[:,:,1] /= b[:,:,2]

np.save("square_100_100_depth", a)
np.save("square_100_100_normal", b)



show(a)
