import numpy as np


def getObjPoints():
    # 调用相机标定参数
    mtx = np.loadtxt('cameraArgs/mtx.txt', delimiter=',')
    # 图像点与时间读取
    ImgPoints = np.loadtxt('data/ImgPoints.txt', delimiter=',')
    # 获取图像点
    imgp = ImgPoints[:, :2]
    # 设置距离参数s
    s = 450
    # 相机内参矩阵、相机旋转矩阵和相机平移矩阵设置
    K = np.mat(mtx)
    R = np.mat(np.eye(3))
    T = np.mat(np.zeros((3, 1)))
    # 分配物理点名称
    objp = []
    for item in imgp:
        item = np.insert(np.mat(item).T, 2, values=1, axis=0)
        itemc = K.I * s * item
        itemw = R.I * (itemc - T)
        itemw = itemw.reshape(1, -1)
        objp.append(itemw.tolist()[0])
    objp = np.mat(objp).reshape(-1, 3)
    x = objp[:, 0]
    y = objp[:, 1]
    t = ImgPoints[:, 2].reshape(-1, 1)
    ObjPoints = np.hstack((x, y, t))
    np.savetxt('data/ObjPoints.txt', ObjPoints, fmt='%f', delimiter=',')


if __name__ == '__main__':
    getObjPoints()
