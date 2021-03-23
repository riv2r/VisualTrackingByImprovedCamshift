import os
from cameraCalib import *

if os.path.exists('cameraArgs/mtx.txt') and os.path.exists('cameraArgs/dist.txt'):
    pass
else:
    cameraCalib()

from camshiftOBJTracking import *
from getImgPoints import *

if os.path.exists('data/ImgPoints.txt'):
    pass
else:
    trackPath = camshiftOBJTracking()
    getImgPoints(trackPath)

from getObjPoints import *

if os.path.exists('data/ObjPoints.txt'):
    pass
else:
    getObjPoints()

import matplotlib.pyplot as plt

ImgPoints = np.loadtxt('data/ImgPoints.txt', delimiter=',')
ObjPoints = np.loadtxt('data/ObjPoints.txt', delimiter=',')
t = ImgPoints[:, 2]
u = ImgPoints[:, 0]
v = ImgPoints[:, 1]
x = ObjPoints[:, 0]
y = ObjPoints[:, 1]


# 计算离散点导数
def calcDiff(x, y):
    diff_x = []
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    k = []
    for i in range(len(diff_y)):
        k.append(diff_y[i] / diff_x[i])

    diff = []
    for i, j in zip(k[0::], k[1::]):
        diff.append((i + j) / 2)
    diff.insert(0, k[0])
    diff.append(k[-1])

    return diff


du = calcDiff(t, u)
dv = calcDiff(t, v)
dx = calcDiff(t, x)
dy = calcDiff(t, y)
ddu = calcDiff(t, du)
ddv = calcDiff(t, dv)
ddx = calcDiff(t, dx)
ddy = calcDiff(t, dy)

plt.subplot(3, 2, 1)
plt.plot(t, u, color='r', linestyle='--')
plt.plot(t, v, color='b', linestyle='--')
plt.xlabel('t(s)')
plt.ylabel('displacement(pixel)')
plt.legend(['u-t', 'v-t'])
plt.title('displacement(pixel)-t')
plt.subplot(3, 2, 2)
plt.plot(t, x, color='r')
plt.plot(t, y, color='b')
plt.xlabel('t(s)')
plt.ylabel('displacement(mm)')
plt.legend(['x-t', 'y-t'])
plt.title('displacement(mm)-t')

plt.subplot(3, 2, 3)
plt.plot(t, du, color='r', linestyle='--')
plt.plot(t, dv, color='b', linestyle='--')
plt.xlabel('t(s)')
plt.ylabel('speed(pixel/s)')
plt.legend(['du-t', 'dv-t'])
plt.title('speed(pixel/s)-t')
plt.subplot(3, 2, 4)
plt.plot(t, dx, color='r')
plt.plot(t, dy, color='b')
plt.xlabel('t(s)')
plt.ylabel('speed(mm/s)')
plt.legend(['dx-t', 'dy-t'])
plt.title('speed(mm/s)-t')

plt.subplot(3, 2, 5)
plt.plot(t, ddu, color='r', linestyle='--')
plt.plot(t, ddv, color='b', linestyle='--')
plt.xlabel('t(s)')
plt.ylabel('acceleration(pixel/s^2)')
plt.legend(['ddu-t', 'ddv-t'])
plt.title('acceleration(pixel/s^2)-t')
plt.subplot(3, 2, 6)
plt.plot(t, ddx, color='r')
plt.plot(t, ddy, color='b')
plt.xlabel('t(s)')
plt.ylabel('acceleration(mm/s^2)')
plt.legend(['ddx-t', 'ddy-t'])
plt.title('acceleration(mm/s^2)-t')

plt.show()
