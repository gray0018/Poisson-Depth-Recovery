from __future__ import division, print_function, absolute_import


from numpy import zeros, save, uint8
from numpy.random import randint


def synthesize(pic_size=256, r=120, per=0.01):
    l = pic_size
    per = int(1/per)
    if 2*r > l:
        raise Exception("r is out of range")
    mask = zeros((l, l), dtype=uint8)
    depth = zeros((l, l))
    normal = zeros((l, l, 3))
    x = []
    y = []
    k = []
    known_index = 0
    for i in range(l):
        for j in range(l):
            if (i-l/2)**2+(j-l/2)**2 <= r**2:
                mask[i][j] = 1
                depth[i][j] = (r**2-(i-l/2)**2-(j-l/2)**2)**0.5
                normal[i][j][0] = i-l/2
                normal[i][j][1] = j-l/2
                normal[i][j][2] = depth[i][j]
                # 这个点已经在圆内了，生成一个随机数，概率1%将该点选为已知点
                if randint(0, per, 1)[0] == 0:
                    x.append(i)
                    y.append(j)
                    k.append(known_index)
                known_index += 1
    save("mask", mask)
    save("known_depth_coordinate_form_001", [x, y])
    save("known_depth_linear_index_form_001", k)
    save("depth", depth)
    save("normal", normal)

synthesize(1024, 500, 0.01)