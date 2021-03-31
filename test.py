import cv2 as cv
'''
HIST=[]

capture=cv.VideoCapture(0)
cv.namedWindow('frame')

while True:
    ret,frame=capture.read()
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    hist=cv.calcHist([hsv],[0],None,[180],[0,180])
    HIST.append(hist)
    cv.imshow('frame',frame)
    key=cv.waitKey(10)
    if key==27:
        break

capture.release()
cv.destroyAllWindows()

hist1=HIST[0]
hist2=HIST[len(HIST)-1]
matchScore=cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)
print(matchScore)
'''

img1=cv.imread('img1.jpg')
img2=cv.imread('img2.jpg')

hsv1=cv.cvtColor(img1,cv.COLOR_BGR2HSV)
hist1=cv.calcHist([hsv1],[0],None,[180],[0,180])

hsv2=cv.cvtColor(img2,cv.COLOR_BGR2HSV)
hist2=cv.calcHist([hsv2],[0],None,[180],[0,180])
matchScore=cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)
print(matchScore)
