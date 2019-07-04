import numpy as np
import scipy.sparse.linalg as lin
from scipy import sparse
from pdrtool import pdr_show


# load data
if True:
    d = np.load("square_100_100_depth.npy")
    n = np.load("square_100_100_normal.npy")
    l = d.shape[0] - 1
    A_row = []
    A_col = []
    A_data = []
    b = []
    w = 1  # the weight for how much we trust the depth map

# build laplacian equation
if True:
    for i in range(1, l):
        for j in range(1, l):
            ind = (i - 1) * (l - 1) + j - 1
            A_row.append(ind)
            A_col.append(ind)
            A_data.append(4)

            laplacian = 4 * d[i][j] - d[i - 1][j] - d[i + 1][j] - d[i][j - 1] - d[i][j + 1]
            div = (n[i + 1][j][0] - n[i - 1][j][0] + n[i][j + 1][1] - n[i][j - 1][1]) / 2
            b.append(div)

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

# add known depth, by default, no just use normal
if False:

    A_row.append(equation_num)
    i = 50
    j = 50
    ind = (i - 1) * (l - 1) + j - 1
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 25
    j = 25
    ind = (i - 1) * (l - 1) + j - 1
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 75
    j = 75
    ind = (i - 1) * (l - 1) + j - 1
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 25
    j = 75
    ind = (i - 1) * (l - 1) + j - 1
    A_col.append(ind)
    A_data.append(1)
    b.append(d[i][j])
    equation_num += 1

    A_row.append(equation_num)
    i = 75
    j = 25
    ind = (i - 1) * (l - 1) + j - 1
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
        tol = 1e-13
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

for i in range(1, l):
    for j in range(1, l):
        ind = (i - 1) * (l - 1) + j - 1
        d[i][j] = res[ind]

zzz = np.load("square_100_100_depth.npy")
zzz = abs(d - zzz)
pdr_show(zzz)
