import numpy as np
import cv2
import glob


def cameraCalib():
    # 棋盘大小
    CHECKERBOARD = (7, 6)
    # 迭代终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 准备对象点
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    # 用于存储所有图像的对象点和图像点的数组。
    objpoints = []  # 真实世界中的3d点
    imgpoints = []  # 图像中的2d点
    images = glob.glob('calibPic\*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到棋盘角落
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        # 如果找到，添加对象点，图像点（细化之后）
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # 绘制并显示拐角
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    '''
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
    '''
    np.savetxt('mtx.txt', mtx, fmt='%f', delimiter=',')
    np.savetxt('dist.txt', dist, fmt='%f', delimiter=',')


if __name__ == '__main__':
    cameraCalib()
