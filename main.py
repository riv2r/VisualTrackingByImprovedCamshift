import os
from cameraCalib import *

'''
需要补充如下文件夹及文件
1.__pycache__:缓存文件夹 内部缓存文件 可自动生成
2..vscode:运行环境配置文件夹
3.calibPics:相机标定素材图片文件夹 内部10张相机标定素材图片
4.chessboard:黑白棋盘格文件夹 内部黑白棋盘word和png版
5.video:标定视频素材文件夹 内部test.mp4
'''

if os.path.exists('cameraArgs/mtx.txt') and os.path.exists('cameraArgs/dist.txt'):
    pass
else:
    cameraCalib()

from OBJTracking import *
from getImgPoints import *

if os.path.exists('data/ImgPoints.txt'):
    pass
else:
    trackPath = OBJTracking()
    getImgPoints(trackPath)