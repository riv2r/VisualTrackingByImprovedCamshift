import cv2

cap=cv2.VideoCapture(1)
cv2.namedWindow('test')
num=0
while True:
    ret, frame=cap.read()
    cv2.imshow('test',frame)
    key=cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('calibPics/'+'calibPics'+str(num)+'.jpg',frame)
        num+=1

cv2.destroyAllWindows()
cap.release()