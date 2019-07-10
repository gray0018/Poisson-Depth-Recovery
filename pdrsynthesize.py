from pdrtool import pdr_synthesize, pdr_normalize_normal_map, pdr_show
from numpy import load, save

pdr_synthesize(1024, 500)

d = load("depth.npy")
# n = pdr_normalize_normal_map(load("normal.npy"))
n = load("normal.npy")

center = d.shape[0]//2
c = center

clip_size = 50

l = clip_size

a = d[c-l:c+l, c-l:c+l]
b = n[c-l:c+l, c-l:c+l]

b[:, :, 0] /= -b[:, :, 2]
b[:, :, 1] /= -b[:, :, 2]
b[:, :, 2] /= -b[:, :, 2]

a += 140

save("square_100_100_depth", a)
save("square_100_100_normal", b)

pdr_show(a)