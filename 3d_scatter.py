from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

normal = np.load("data/normal.npy")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = normal[:,:,0].reshape(1, -1)
ys = normal[:,:,1].reshape(1, -1)
zs = normal[:,:,2].reshape(1, -1)

ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
