import cv2
import numpy as np
import time

# 调用相机标定参数
mtx = np.loadtxt('mtx.txt', delimiter=',')
dist = np.loadtxt('dist.txt', delimiter=',')

# 初始化全局变量
xs, ys, ws, hs = 0, 0, 0, 0  # 初始化鼠标选取区域左上角坐标和长宽
xo, yo = 0, 0  # 初始化左上角原始坐标
selectObject = False  # 标记是否选取了目标
trackObject = 0  # 1 表示有追踪对象 0 表示无追踪对象 -1 表示追踪对象尚未计算 Camshift 所需的属性
trackPath = []  # 轨迹路径列表


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
def camshiftOBJTracking():
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject, trackPath

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 捕获摄像头
    cv2.namedWindow('CamshiftOBJTracking')
    cv2.setMouseCallback('CamshiftOBJTracking', onMouse)  # 调用鼠标回调函数
    # 设置结束标志，10 次迭代或至少 1 次移动
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while (True):
        ret, frame = cap.read()  # 连续读取视频帧
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # 校准图像
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # 裁剪图像
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将BGR转换为HSV空间
        mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))  # 制作掩模版
        
        # 表示已捕获跟踪对象
        if trackObject != 0:
            if trackObject == -1:
                track_window = (xs, ys, ws, hs)
                mask_roi = mask[ys:ys + hs, xs:xs + ws]
                hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
                # 构建图像分布直方图
                roi_hist = cv2.calcHist([hsv_roi], [0], mask_roi, [181], [0, 180])
                # 归一化处理
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                trackObject = 1
            # 反向投影
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            dst &= mask
            # 调用CAMshift算法
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            x, y, w, h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 记录时间
            now = time.time()
            trackPath.append([x + w // 2, y + h // 2, now])

            for i in range(1, len(trackPath)):
                cv2.line(frame, (trackPath[i - 1][0], trackPath[i - 1][1]), (trackPath[i][0], trackPath[i][1]),
                         (0, 255, 0),
                         2)

        if selectObject and ws > 0 and hs > 0:
            cv2.bitwise_not(frame[ys:ys + hs, xs:xs + ws], frame[ys:ys + hs, xs:xs + ws])

        cv2.imshow('CamshiftOBJTracking', frame)

        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    return trackPath


if __name__ == '__main__':
    camshiftOBJTracking()
