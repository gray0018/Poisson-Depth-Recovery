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
    print("usage: python main.py demo_name(sphere or bunny) projection_model(-o or -p)")


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


def orthographic_bunny():
    normal = np.load("data/bunny/bunny_normal.npy")
    depth_gt = np.load("data/bunny/bunny_depth.npy")
    mask = ~(depth_gt == 0)
    mask = erode_mask(erode_mask(mask))  # remove boundary
    normal[..., 2] = -normal[..., 2]
    normal[..., 2][~mask] = -1

    depth_gt = 10 - depth_gt  # subtract from 10, translate position to depth

    # add 10 depth info manually
    depth_info = np.zeros_like(depth_gt)
    depth_info[depth_gt.shape[0] // 2, depth_gt.shape[1] // 2] = depth_gt[
        depth_gt.shape[0] // 2, depth_gt.shape[1] // 2]
    depth_info[323, 94] = depth_gt[323, 94]
    depth_info[182, 332] = depth_gt[182, 332]
    depth_info[135, 157] = depth_gt[135, 157]
    depth_info[830, 160] = depth_gt[830, 160]
    depth_info[746, 230] = depth_gt[746, 230]
    depth_info[218, 185] = depth_gt[218, 185]
    depth_info[810, 230] = depth_gt[810, 230]
    depth_info[664, 163] = depth_gt[664, 163]
    depth_info[234, 185] = depth_gt[234, 185]

    # delta is the scaling in orthographic projection
    delta = 1 / 256
    normal[..., 0] = -normal[..., 0] / normal[..., 2] * delta
    normal[..., 1] = -normal[..., 1] / normal[..., 2] * delta

    # in OpenCV mask should be int type, 0.05 is the weight for depth fusion
    test = PoissonOperator(normal[..., :2], mask.astype(np.int8), depth_info, 0.05)
    depth_est = test.run()

    mask = erode_mask(erode_mask(mask))  # remove boundary
    depth_est[~mask] = np.nan
    depth_gt[~mask] = np.nan

    plot_result(depth_est, depth_gt)


def orthographic_sphere():
    normal = np.load("data/sphere/sphere_normal.npy")
    depth_gt = np.load("data/sphere/sphere_depth.npy")

    mask = ~(depth_gt == 0)
    mask = erode_mask(erode_mask(mask))  # remove boundary
    normal[..., 2] = -normal[..., 2]
    normal[..., 2][~mask] = -1

    depth_gt = 10 - depth_gt  # subtract from 10, translate position to depth

    # add one depth info manually
    depth_info = np.zeros_like(depth_gt)
    depth_info[depth_gt.shape[0] // 2, depth_gt.shape[1] // 2] = depth_gt[
        depth_gt.shape[0] // 2, depth_gt.shape[1] // 2]

    # delta is the scaling in orthographic projection
    delta = 1 / 512
    normal[..., 0] = -normal[..., 0] / normal[..., 2] * delta
    normal[..., 1] = -normal[..., 1] / normal[..., 2] * delta

    # in OpenCV mask should be int type, 0.01 is the weight for depth fusion
    test = PoissonOperator(normal[..., :2], mask.astype(np.int8), depth_info, 0.1)
    depth_est = test.run()

    depth_est[~mask] = np.nan
    depth_gt[~mask] = np.nan

    plot_result(depth_est, depth_gt)


def perspective_sphere():
    normal = np.load("data/sphere-perspective/sphere_perspective_normal.npy")
    depth_gt = np.load("data/sphere-perspective/sphere_perspective_depth.npy")

    mask = ~(depth_gt < 0)
    mask = erode_mask(erode_mask(mask))

    # align coordinate
    normal[..., 0] = -normal[..., 0]

    normal[..., 0][~mask] = 0
    normal[..., 1][~mask] = 0
    normal[..., 2][~mask] = 1

    n = normal
    d = depth_gt
    f = 85 * d.shape[1] / 36

    x, y = np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0]))
    u = x - d.shape[1] // 2
    v = np.flipud(y) - d.shape[0] // 2

    p_ = -n[..., 0] / (u * n[..., 0] + v * n[..., 1] + f * n[..., 2])
    q_ = -n[..., 1] / (u * n[..., 0] + v * n[..., 1] + f * n[..., 2])
    d_ = np.log(d)

    # add one depth info manually
    depth_info = np.zeros_like(d_)
    depth_info[d_.shape[0] // 2, d_.shape[1] // 2] = d_[d_.shape[0] // 2, d_.shape[1] // 2]

    # in OpenCV mask should be int type, 0.01 is the weight for depth fusion
    test = PoissonOperator(np.dstack([p_, q_]), mask.astype(np.int8), depth_info, 0.1)
    depth_est = np.exp(test.run())

    depth_est[~mask] = np.nan
    depth_gt[~mask] = np.nan

    plot_result(depth_est, depth_gt)


def perspective_bunny():
    pass

def read_camera(path):
    camera = np.loadtxt(path)
    if camera[2, 2] == 0:
        # orthographic camera
        Zc = camera[2, 3]


if __name__ == '__main__':

    n = np.load("data/sphere/normal.npy")
    n_mask = np.load("data/sphere/normal_mask.npy")
    n[..., 2][~n_mask] = -1
    

    d = np.load("data/sphere/depth.npy")
    d_mask = np.load("data/sphere/depth_mask.npy")
    d[~d_mask] = 0

    camera = np.loadtxt(path)

    if camera[2, 2] == 0:
        # orthographic camera
        Zc = camera[2, 3]
        p = -n[..., 0] / n[..., 2] / Zc
        q = -n[..., 1] / n[..., 2] / Zc
        batch = PoissonOperator(np.dstack([p, q]), n_mask.astype(np.int8), d, 0.1)
        d_est = batch.run()

        plt.figure()
        plt.imshow(d_est, "gray")
        plt.title("Estimated Depth Map")
        plt.axis('off')
        plt.colorbar()
        plt.show()
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
        d_ = np.log(d)

        batch = PoissonOperator(np.dstack([p, q]), n_mask.astype(np.int8), d_, 0.1)
        d_est = np.exp(test.run())

        plt.figure()
        plt.imshow(d_est, "gray")
        plt.title("Estimated Depth Map")
        plt.axis('off')
        plt.colorbar()
        plt.show()