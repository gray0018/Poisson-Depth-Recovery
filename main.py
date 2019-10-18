import sys
import numpy as np
import matplotlib.pyplot as plt
from lib.poisson import PoissonOperator


def plot_result(depth_est, depth_gt):
    plt.figure()

    plt.subplot(131)
    plt.imshow(depth_est, "gray")
    plt.title("Estimated Depth Map")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(depth_gt, "gray")
    plt.title("Ground Truth")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(np.abs(depth_gt - depth_est), "gray")
    plt.title("Error Map")
    plt.axis('off')
    plt.colorbar()

    plt.show()


def usage():
    print("Orthographic example: python main.py -o normal.npy mask_normal.npy depth.npy mask_depth.npy camera.ini")
    print("Projective example: python main.py -o normal.npy mask_normal.npy depth.npy mask_depth.npy camera.ini")


def erode_mask(mask):
    new_mask = np.zeros_like(mask)
    for i in range(1, mask.shape[0]-1):
        for j in range(1, mask.shape[1]-1):
            if mask[i+1,j] and mask[i-1,j] and mask[i,j-1] and mask[i,j+1]:
                new_mask[i, j] = 1
    return new_mask.astype(np.bool_)


def plot_result(depth_est, depth_gt):
    plt.figure()

    plt.subplot(131)
    plt.imshow(depth_est, "gray")
    plt.title("Estimated Depth Map")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(depth_gt, "gray")
    plt.title("Ground Truth")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(np.abs(depth_gt - depth_est), "gray")
    plt.title("Error Map")
    plt.axis('off')
    plt.colorbar()

    plt.show()


def plot_est_result(d_est):
    plt.figure()
    plt.imshow(d_est, "gray")
    plt.title("Estimated Depth Map")
    plt.axis('off')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':

    dir = sys.argv[1]

    n = np.load(dir+"/normal.npy")
    n_mask = np.load(dir+"/normal_mask.npy")
    n[..., 2][~n_mask] = -1

    d = np.load(dir+"/depth.npy")
    d_mask = np.load(dir+"/depth_mask.npy")

    camera = np.loadtxt(dir+"/camera.ini")

    if camera[2, 2] == 0:
        # orthographic camera
        Zc = camera[2, 3]
        p = -n[..., 0] / n[..., 2] / Zc
        q = -n[..., 1] / n[..., 2] / Zc
        d[~d_mask] = 0

        batch = PoissonOperator(np.dstack([p, q]), n_mask.astype(np.int8), d, 0.1)
        d_est = batch.run()

        d_est[~n_mask] = np.nan
        d_gt = np.load(dir+"/depth.npy")
        d_gt[~n_mask] = np.nan
        plot_result(d_est, d_gt)
    else:
        # perspective camera
        fx = camera[0, 0]
        fy = camera[1, 1]
        px = camera[0, 2]
        py = camera[1, 2]

        x, y = np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0]))
        u = x - px
        v = np.flipud(y) - py

        p = -n[..., 0] / (u * n[..., 0] + v * n[..., 1] + fx * n[..., 2])
        q = -n[..., 1] / (u * n[..., 0] + v * n[..., 1] + fy * n[..., 2])
        d[~d_mask] = 1
        d = np.log(d)

        batch = PoissonOperator(np.dstack([p, q]), n_mask.astype(np.int8), d, 0.1)
        d_est = np.exp(batch.run())

        d_est[~n_mask] = np.nan
        d_gt = np.load(dir+"/depth.npy")
        d_gt[~n_mask] = np.nan
        plot_result(d_est, d_gt)


