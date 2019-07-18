import OpenEXR
import Imath
import numpy as np
import cv2
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

# pic = np.concatenate((channels["shNormal.B"][:, :, None], channels["shNormal.G"][:, :, None], channels["shNormal.R"][:, :, None]), axis=-1)
# cv2.imshow(None, pic/2+.5)
# cv2.waitKey()
# cv2.destroyAllWindows()
#

normal = np.concatenate((channels["shNormal.R"][:, :, None], channels["shNormal.G"][:, :, None], channels["shNormal.B"][:, :, None]), axis=-1)
fig1 = plt.subplot(1, 2, 1)
fig1 = plt.imshow(normal/2+.5)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)

depth = channels["distance.Y"]
fig2 = plt.subplot(1, 2, 2)
fig2 = plt.imshow(depth)
fig2.axes.get_xaxis().set_visible(False)
fig2.axes.get_yaxis().set_visible(False)

plt.show()
