import cv2

vdop = "video/test1.mp4"#输入视频路径
cap = cv2.VideoCapture(vdop) 
fps = cap.get(cv2.CAP_PROP_FPS) #获取输入视频的帧率
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))#获取输入视频的大小
fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  #These are the FOURCCs to compressed formats

out_path = "video/chtest1.mp4" #输出2倍速的avi格式的视频路径
output_viedo = cv2.VideoWriter()
fps = 0.05*fps #2倍速处理
#isColor：如果该位值为Ture，解码器会进行颜色框架的解码，否则会使用灰度进行颜色架构
output_viedo.open(out_path , fourcc, fps, size, isColor=True)
rval = True
while rval:
    rval, img = cap.read()#逐帧读取原视频
    output_viedo.write(img)#写入视频帧
output_viedo.release()
cap.release()