import cv2
import numpy as np
import time

from numpy.random import gamma

# 调用相机标定参数
mtx = np.loadtxt('cameraArgs/mtx.txt', delimiter=',')
dist = np.loadtxt('cameraArgs/dist.txt', delimiter=',')

# 初始化鼠标选取区域左上角坐标和长宽
xs, ys, ws, hs = 0, 0, 0, 0
# 初始化左上角原始坐标
xo, yo = 0, 0
# 标记是否选取了目标
selectObject = False
# 1 表示有追踪对象 0 表示无追踪对象 -1 表示追踪对象尚未计算 Camshift 所需的属性
trackObject = 0
# 轨迹路径列表
trackPath = []

# kalman滤波器初始化
# 4 状态数 包括(x,y,dx,dy)坐标及速度(每次移动的距离) 2 观测量 表示能测量到的值
kalman = cv2.KalmanFilter(4, 2)
# 转移矩阵A
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
# 过程噪声协方差矩阵Q
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 1e-5
# 测量矩阵H
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
# 测量噪声协方差矩阵R
kalman.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], np.float32) * 1e-1
# 最优估计值statePost
kalman.statePost = np.float32(np.random.randn(4, 1))
# 最小均方误差矩阵P
kalman.errorCovPost = np.ones((4, 4), np.float32)


# 鼠标回调函数
def onMouse(event, x, y, flags, prams):
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject, trackPath
    if selectObject == True:
        xs = min(x, xo)
        ys = min(y, yo)
        ws = abs(x - xo)
        hs = abs(y - yo)
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        trackObject = -1
    elif event == cv2.EVENT_RBUTTONDOWN:
        xs, ys, ws, hs = 0, 0, 0, 0
        xo, yo = 0, 0
        selectObject = False
        trackObject = 0
        trackPath = []


# 处理算法函数
def OBJTracking():
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject, trackPath
    # 捕获摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow('CamshiftOBJTracking')
    # 调用鼠标回调函数
    cv2.setMouseCallback('CamshiftOBJTracking', onMouse)
    # 设置结束标志，10 次迭代或至少 1 次移动
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # 采用KNN背景分割
    backSub = cv2.createBackgroundSubtractorKNN()

    while (True):
        # 连续读取视频帧
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # 校准图像
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # 裁剪图像
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
        # 镜像视频帧
        frame = cv2.flip(frame, 90)
        # 将BGR转换为HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 制作掩模版
        mask = cv2.inRange(hsv, np.array((0., 43., 46.)), np.array((180., 255., 255.)))
        # 增强对比度
        b, g, r = cv2.split(frame)
        b1 = cv2.equalizeHist(b)
        g1 = cv2.equalizeHist(g)
        r1 = cv2.equalizeHist(r)
        frame = cv2.merge([b1, g1, r1])

        # 表示已捕获跟踪对象
        if trackObject != 0:
            # 计算camshift所需属性
            if trackObject == -1:
                track_window = (xs, ys, ws, hs)
                roi_mask = mask[ys:ys + hs, xs:xs + ws]
                roi_hsv = hsv[ys:ys + hs, xs:xs + ws]
                # 构建图像分布直方图
                roi_hist = cv2.calcHist([roi_hsv], [0], roi_mask, [180], [0, 180])
                # 归一化处理
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                trackObject = 1
            # 制作背景分割掩模版 并应用于当前帧
            fgMask = backSub.apply(frame)
            backSubFrame = cv2.bitwise_and(frame, frame, mask = fgMask)
            backSubHsv = cv2.cvtColor(backSubFrame, cv2.COLOR_BGR2HSV)
            # 反向投影
            backProj = cv2.calcBackProject([backSubHsv], [0], roi_hist, [0, 180], 1)
            backProj &= mask

            # 调用CAMshift算法
            ret, track_window = cv2.CamShift(backProj, track_window, term_crit)
            x, y, w, h = track_window
            # 卡尔曼滤波_预测
            statePre = kalman.predict()
            # 卡尔曼滤波_更新
            measurement = np.array([[x + w // 2],
                                    [y + h // 2]], np.float32)
            kalman.correct(measurement)
            # 获取校正后的位置坐标
            px, py = int(kalman.statePost[0]), int(kalman.statePost[1])
            # 绘制捕获方框
            cv2.rectangle(frame, (px - w // 2, py - h // 2), (px + w // 2, py + h // 2), (255, 0, 0), 2)
            # 记录时间
            now = time.time()
            trackPath.append([px, py, now])

            for i in range(1, len(trackPath)):
                cv2.line(frame, (trackPath[i - 1][0], trackPath[i - 1][1]), (trackPath[i][0], trackPath[i][1]),
                         (0, 255, 0), 2)

        # 显示鼠标选择区域
        if selectObject and ws > 0 and hs > 0:
            cv2.bitwise_not(frame[ys:ys + hs, xs:xs + ws], frame[ys:ys + hs, xs:xs + ws])

        # 显示图像
        cv2.imshow('CamshiftOBJTracking', frame)

        # 控制程序停止
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    return trackPath


if __name__ == '__main__':
    OBJTracking()
