import sys
import numpy as np
import matplotlib.pyplot as plt
from lib.poisson import PoissonOperator
from lib.poisson import erode_mask


def bunny():
    normal = np.load("data/bunny/bunny_normal.npy")
    depth_gt = np.load("data/bunny/bunny_depth.npy")
    mask = ~(depth_gt == 0)
    mask = erode_mask(erode_mask(mask))  # remove boundary
    normal[..., -1][~mask] = 1
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

    plt.figure()

    plt.subplot(131)
    plt.imshow(depth_est, "gray")
    plt.title("Estimated Depth Map")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(depth_est, "gray")
    plt.title("Ground Truth")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(np.abs(depth_gt - depth_est), "gray")
    plt.title("Error Map")
    plt.axis('off')
    plt.colorbar()

    plt.show()


def sphere():
    normal = np.load("data/sphere/sphere_normal.npy")
    depth_gt = np.load("data/sphere/sphere_depth.npy")

    mask = ~(depth_gt == 0)
    mask = erode_mask(erode_mask(mask))  # remove boundary
    normal[..., -1][~mask] = 1

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

    plt.figure()

    plt.subplot(131)
    plt.imshow(depth_est, "gray")
    plt.title("Estimated Depth Map")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(depth_est, "gray")
    plt.title("Ground Truth")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(np.abs(depth_gt - depth_est), "gray")
    plt.title("Error Map")
    plt.axis('off')
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    if sys.argv[1] == "bunny":
        bunny()
    elif sys.argv[1] == "sphere":
        sphere()
    else:
        print("usage: python main.py demo_name(sphere or bunny)")
