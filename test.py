import cv2

cap=cv2.VideoCapture(0)
cv2.namedWindow('test')

while True:
    ret, frame = cap.read()
    if cv2.waitKey(10)==27:
        break
    cv2.imshow('test',frame)

cap.release()
cv2.destroyAllWindows()