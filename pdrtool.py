from __future__ import division, print_function, absolute_import


from numpy import zeros, save, uint8, reshape
from numpy.random import randint

from sklearn.preprocessing import normalize

from matplotlib.pyplot import figure, subplot, show


def pdr_show(pic, color="gray", name="Picture", fontdict={'fontsize': 27}):

    figure(figsize=(10,10))
    fig = subplot(1,1,1)
    fig.imshow(pic, color)
    fig.set_title(name, fontdict)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    show()


def pdr_normalize_normal_map(N):
    """
    N is a unnormalized normal map of shape H_W_3. Normalize N across the third dimension.
    :param N:
    :return:
    """
    h, w, c = N.shape
    N = reshape(N, (-1, c))
    N = normalize(N, axis=1)
    N = reshape(N, (h, w, c))
    return N


def pdr_synthesize(pic_size=256, r=120, per=0.01):
    l = pic_size
    per = int(1/per)
    if 2*r > l:
        raise Exception("r is out of range")

    depth = zeros((l, l))
    normal = zeros((l, l, 3))
    for i in range(l):
        for j in range(l):
            if (i-l/2)**2+(j-l/2)**2 <= r**2:
                depth[i][j] = (r**2-(i-l/2)**2-(j-l/2)**2)**0.5
                normal[i][j][0] = i-l/2
                normal[i][j][1] = j-l/2
                normal[i][j][2] = depth[i][j]
                # 这个点已经在圆内了，生成一个随机数，概率1%将该点选为已知点

    save("depth", depth)
    save("normal", normal)
