import numpy as np


def getImgPoints(trackPath):
    trackPath = np.mat(trackPath).reshape(-1, 3)
    trackPath[:, 2] = trackPath[:, 2] - trackPath[0, 2]
    np.savetxt('ImgPoints.txt', trackPath, fmt='%f', delimiter=',')


if __name__ == '__main__':
    getImgPoints()
