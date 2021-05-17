import numpy as np


def getImgPoints(trackPath):
    trackPath = np.mat(trackPath).reshape(-1, 3)
    np.savetxt('data/ImgPoints.txt', trackPath, fmt='%f', delimiter=',')


if __name__ == '__main__':
    getImgPoints()
