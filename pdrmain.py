import numpy as np
import scipy.sparse.linalg as lin
from scipy import sparse
from pdrtool import pdr_show
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

def split_channel(f, channel, float_flag=True):
    dw = f.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    if float_flag:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
    else:
        pt = Imath.PixelType(Imath.PixelType.HALF)
    channel_str = f.channel(channel, pt)
    img = np.frombuffer(channel_str, dtype=np.float32)
    img.shape = (size[1], size[0])
    return img


f = OpenEXR.InputFile("bunny.exr")

channels = dict()
for channel_name in f.header()["channels"]:
    print(channel_name)
    split_channel(f, channel_name)
    channels[channel_name] = split_channel(f, channel_name)

n = np.concatenate((channels["shNormal.R"][:, :, None], channels["shNormal.G"][:, :, None], channels["shNormal.B"][:, :, None]), axis=-1)
d = channels["distance.Y"]
gt = d[300:400, 300:400]

n = n[300:400, 300:400, :].copy()
d = d[300:400, 300:400].copy()

if False:
    fig1 = plt.subplot(1, 2, 1)
    fig1 = plt.imshow(n / 2 + .5)
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    depth = channels["distance.Y"]
    fig2 = plt.subplot(1, 2, 2)
    fig2 = plt.imshow(gt, "gray")
    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)

    plt.show()

n[:, :, 0] /= -n[:, :, 2]
n[:, :, 1] /= -n[:, :, 2]
n[:, :, 2] /= -n[:, :, 2]

# load data
if True:
    # d = np.load("square_100_100_depth.npy")
    # n = np.load("square_100_100_normal.npy")
    l = d.shape[0]
    A_row = []
    A_col = []
    A_data = []
    b = []
    w = 1  # the weight for how much we trust the depth map

# build Laplacian equation with boundary substitution
if False:
    for i in range(1, l):
        for j in range(1, l):
            ind = (i - 1) * (l - 1) + j - 1
            A_row.append(ind)
            A_col.append(ind)
            A_data.append(4)

            laplacian = 4 * d[i][j] - d[i - 1][j] - d[i + 1][j] - d[i][j - 1] - d[i][j + 1]
            div = -(n[i + 1][j][0] - n[i - 1][j][0] + n[i][j + 1][1] - n[i][j - 1][1]) / 2
            b.append(div)
            error_tmp = abs(laplacian - div)

            if i - 1 == 0:  # up
                b[-1] += d[i - 1][j]
            else:
                A_row.append(ind)
                A_col.append(ind - (l - 1))
                A_data.append(-1)

            if i + 1 == l:  # down
                b[-1] += d[i + 1][j]
            else:
                A_row.append(ind)
                A_col.append(ind + (l - 1))
                A_data.append(-1)

            if j - 1 == 0:  # left
                b[-1] += d[i][j - 1]
            else:
                A_row.append(ind)
                A_col.append(ind - 1)
                A_data.append(-1)

            if j + 1 == l:  # right
                b[-1] += d[i][j + 1]
            else:
                A_row.append(ind)
                A_col.append(ind + 1)
                A_data.append(-1)

    equation_num = len(b)
    variable_num = len(b)

# build Laplacian equation without boundary substitution
if True:
    for i in range(0, l):
        for j in range(0, l):
            ind = i * l + j
            if i == 0 and j == 0:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-2)

                A_row.append(ind)
                A_col.append(ind+1)
                A_data.append(1)
                # normal part 1
                np1 = (n[i][j][1] + n[i][j+1][1]) / 2

                A_row.append(ind)
                A_col.append(ind+l)
                A_data.append(1)
                np2 = (n[i][j][0] + n[i + 1][j][0])/2

                b.append(np1+np2)
            elif i == 0 and j == l - 1:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-2)

                A_row.append(ind)
                A_col.append(ind - 1)
                A_data.append(1)
                # normal part 1
                np1 = -(n[i][j][1] + n[i][j - 1][1]) / 2

                A_row.append(ind)
                A_col.append(ind + l)
                A_data.append(1)
                np2 = (n[i][j][0] + n[i + 1][j][0])/2

                b.append(np1+np2)
            elif i == l - 1 and j == 0:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-2)

                A_row.append(ind)
                A_col.append(ind + 1)
                A_data.append(1)
                # normal part 1
                np1 = (n[i][j][1] + n[i][j + 1][1]) / 2

                A_row.append(ind)
                A_col.append(ind - l)
                A_data.append(1)
                np2 = -(n[i][j][0] + n[i - 1][j][0]) / 2

                b.append(np1 + np2)
            elif i == l - 1 and j == l - 1:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-2)

                A_row.append(ind)
                A_col.append(ind - 1)
                A_data.append(1)
                # normal part 1
                np1 = -(n[i][j][1] + n[i][j - 1][1]) / 2

                A_row.append(ind)
                A_col.append(ind - l)
                A_data.append(1)
                np2 = -(n[i][j][0] + n[i - 1][j][0]) / 2

                b.append(np1 + np2)
            elif i == 0:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-3)

                A_row.append(ind)
                A_col.append(ind + 1)
                A_data.append(1)
                # normal part 1
                np1 = (n[i][j][1] + n[i][j + 1][1]) / 2

                A_row.append(ind)
                A_col.append(ind + l)
                A_data.append(1)
                np2 = (n[i][j][0] + n[i + 1][j][0]) / 2

                A_row.append(ind)
                A_col.append(ind - 1)
                A_data.append(1)
                # normal part 3
                np3 = -(n[i][j][1] + n[i][j - 1][1]) / 2

                b.append(np1 + np2 + np3)
            elif i == l-1:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-3)

                A_row.append(ind)
                A_col.append(ind + 1)
                A_data.append(1)
                # normal part 1
                np1 = (n[i][j][1] + n[i][j + 1][1]) / 2

                A_row.append(ind)
                A_col.append(ind - l)
                A_data.append(1)
                np2 = -(n[i][j][0] + n[i - 1][j][0]) / 2

                A_row.append(ind)
                A_col.append(ind - 1)
                A_data.append(1)
                # normal part 3
                np3 = -(n[i][j][1] + n[i][j - 1][1]) / 2

                b.append(np1 + np2 + np3)
            elif j == 0:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-3)

                A_row.append(ind)
                A_col.append(ind + 1)
                A_data.append(1)
                # normal part 1
                np1 =  (n[i][j][1] + n[i][j + 1][1]) / 2

                A_row.append(ind)
                A_col.append(ind - l)
                A_data.append(1)
                np2 = -(n[i][j][0] + n[i - 1][j][0]) / 2

                A_row.append(ind)
                A_col.append(ind + l)
                A_data.append(1)
                # normal part 3
                np3 =  (n[i][j][0] + n[i + 1][j][0]) / 2

                b.append(np1 + np2 + np3)
            elif j == l-1:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-3)

                A_row.append(ind)
                A_col.append(ind - 1)
                A_data.append(1)
                # normal part 1
                np1 = -(n[i][j][1] + n[i][j - 1][1]) / 2

                A_row.append(ind)
                A_col.append(ind - l)
                A_data.append(1)
                np2 = -(n[i][j][0] + n[i - 1][j][0]) / 2

                A_row.append(ind)
                A_col.append(ind + l)
                A_data.append(1)
                np3 =  (n[i][j][0] + n[i + 1][j][0]) / 2

                b.append(np1 + np2 + np3)
            else:
                A_row.append(ind)
                A_col.append(ind)
                A_data.append(-4)

                laplacian = -4 * d[i][j] + d[i - 1][j] + d[i + 1][j] + d[i][j - 1] + d[i][j + 1]
                div = (n[i + 1][j][0] - n[i - 1][j][0] + n[i][j + 1][1] - n[i][j - 1][1]) / 2
                b.append(div)
                error_tmp = abs(laplacian - div)

                A_row.append(ind)
                A_col.append(ind - l)
                A_data.append(1)

                A_row.append(ind)
                A_col.append(ind + l)
                A_data.append(1)

                A_row.append(ind)
                A_col.append(ind - 1)
                A_data.append(1)

                A_row.append(ind)
                A_col.append(ind + 1)
                A_data.append(1)

    equation_num = len(b)
    variable_num = len(b)

# add known depth, by default, no just use normal
if True:
    A_row.append(equation_num)
    i = 5
    j = 98
    ind = i * l + j
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 50
    j = 75
    ind = i * l + j
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 50
    j = 50
    ind = i * l + j
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 25
    j = 25
    ind = i * l + j
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 75
    j = 75
    ind = i * l + j
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 75
    j = 25
    ind = i * l + j
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 25
    j = 75
    ind = i * l + j
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

# 解方程
if True:
    A_row = np.array(A_row)
    A_col = np.array(A_col)
    A_data = np.array(A_data)
    b = np.array(b)
    A = sparse.csr_matrix((A_data, (A_row, A_col)), shape=(equation_num, variable_num))

    # here choose method to solve the equation, default use lsqr with tol=10^-13
    if True:
        tol = 1e-15
        res = lin.lsqr(A, b, atol=tol, btol=tol, conlim=1 / tol)[0]

        # res = lin.inv(A)*b

        # res = lin.spsolve(A, b)        # Top 1

        # res = lin.cgs(A, b)[0]

        # res = lin.gmres(A, b)[0]

        # res = lin.lgmres(A, b)[0]

        # res = lin.qmr(A, b)[0]

        # res = lin.gcrotmk(A, b)[0]

        # res = lin.cg(A, b)[0]
        # res = lin.bicg(A, b)[0]
        # res = lin.bicgstab(A, b)[0]

        # res = lin.lsmr(A, b)[0]

        # res = lin.minres(A, b)[0]

# 替换计算出的depth

for i in range(0, l):
    for j in range(0, l):
        ind = i*l+j
        d[i][j] = res[ind]

fig1 = plt.subplot(1, 2, 1)
fig1 = plt.imshow(gt, "gray")
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
depth = channels["distance.Y"]
fig2 = plt.subplot(1, 2, 2)
fig2 = plt.imshow(d, "gray")
fig2.axes.get_xaxis().set_visible(False)
fig2.axes.get_yaxis().set_visible(False)

plt.show()
