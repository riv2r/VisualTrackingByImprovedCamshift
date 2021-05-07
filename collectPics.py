import cv2

def collectPics(num):
    # 使用外置摄像头
    cap=cv2.VideoCapture(1)
    cv2.namedWindow('test')
    num+=1
    while True:
        ret, frame=cap.read()
        cv2.imshow('test',frame)
        key=cv2.waitKey(10)
        if key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite('calibPics/'+'pic'+str(num)+'.jpg',frame)
            num+=1

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    collectPics()