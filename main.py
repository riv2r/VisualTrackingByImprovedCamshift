import os
from cameraCalib import *

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