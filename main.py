# from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
# from scipy.sparse.linalg import spsolve


d = np.load("depth.npy")
n = np.load("normal.npy")
a = 1
4*d[128][128]-d[127][128]-d[129][128]-d[128][127]-d[128][129]

# normal = np.load("normal.npy")
# normal = np.rot90(normalize_normal_map(normal)/2+.5, 1)
# show(normal, color=None)


if True:
    if True:
        make_mask()
        mask = np.load("mask.npy")
        known_depth_linear = np.load("known_depth_linear_index_form_001.npy")
        known_depth_coor = np.load("known_depth_coordinate_form_001.npy")
        depth = np.load("depth.npy")
        normal = np.load("normal.npy")
        normal = normalize_normal_map(normal)

        t = np.where(mask)
        mask_coor = np.array((t[0], t[1])).T
        n = len(t[0])

        k_l = known_depth_linear
        k_c = np.array((known_depth_coor[0], known_depth_coor[1])).T

        A_row = []
        A_col = []
        A_data = []
        b = []
        w = 1  # the weight for how much we trust the depth map

        d = {}  # map_2dpixel_in_mask_to_linear_index

    for i in range(n):
        x = mask_coor[i][0]
        y = mask_coor[i][1]
        # mask 中的 (x, y) 按照线性坐标排序是第 i个点
        # 因为此处是圆所以需要把每个（x，y）形式坐标线性索引化
        # 在构建线性方程时才能把每个像素对应到某个变量
        d[(x, y)] = i

    #
    for i in range(n):
        x = mask_coor[i][0]
        y = mask_coor[i][1]
        # 只考虑相邻的四个点都在内侧的情况
        if (x, y - 1) in d and (x, y + 1) in d and (x - 1, y) in d and (x + 1, y) in d:
            if i in k_l:  # if the i-th pixel's depth is known
                A_row.append(i)
                A_col.append(i)
                A_data.append(w)
                b.append(w * depth[x][y])
            else:
                # if not(i in k and i-1 in k and i+1 in k and d[(x-1, y)] in k and d[(x+1, y)] in k):
                A_row.append(i)
                A_col.append(i)
                A_data.append(4)
                ss = 0
                if normal[x+1][y][2]:
                    ss += normal[x+1][y][0]/normal[x+1][y][2]
                else:
                    ss += normal[x + 1][y][0]
                if normal[x-1][y][2]:
                    ss -= normal[x-1][y][0]/normal[x-1][y][2]
                else:
                    ss -= normal[x - 1][y][0]
                if normal[x][y+1][2]:
                    ss += normal[x][y+1][1]/normal[x][y+1][2]
                else:
                    ss += normal[x][y + 1][1]
                if normal[x][y-1][2]:
                    ss -= normal[x][y-1][1]/normal[x][y-1][2]
                else:
                    ss -= normal[x][y - 1][1]
                b.append(ss/2)
                # 分别考察四个相邻点的深度是否已知
                left, right, up, down = i - 1, i + 1, d[(x - 1, y)], d[(x + 1, y)]

                if left in k_l:
                    b[-1] += depth[x][y - 1]
                else:
                    A_row.append(i)
                    A_col.append(i - 1)
                    A_data.append(-1)

                if right in k_l:
                    b[-1] += depth[x][y + 1]
                else:
                    A_row.append(i)
                    A_col.append(i + 1)
                    A_data.append(-1)

                if up in k_l:
                    b[-1] += depth[x - 1][y]
                else:
                    A_row.append(i)
                    A_col.append(d[(x - 1, y)])
                    A_data.append(-1)

                if down in k_l:
                    b[-1] += depth[x + 1][y]
                else:
                    A_row.append(i)
                    A_col.append(d[(x + 1, y)])
                    A_data.append(-1)
        else:
            b.append(0)

    # 解方程
    if True:
        A_row = np.array(A_row)
        A_col = np.array(A_col)
        A_data = np.array(A_data)
        b = np.array(b)
        A = sparse.csr_matrix((A_data, (A_row, A_col)), shape=(n, n))

        res_all = cg(A, b)
        res = res_all[0]

    #替换计算出的depth
    for i in range(n):
        x = mask_coor[i][0]
        y = mask_coor[i][1]
        if (x - 1, y) in d and (x + 1, y) in d and (x, y - 1) in d and (x, y + 1) in d:
            # if not(i in k_l):
            depth[x][y] = res[i]
    zzz = np.load("depth.npy")
    show(abs(depth-zzz))
