import cv2
from sklearn.metrics import roc_curve, auc


# 检测
# 识别
# adaboost
def detect(img, cascade):
    # 进行人脸的多尺度检测
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, flags=cv2.CASCADE_DO_CANNY_PRUNING,
                                     minSize=(3, 3))
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


# #rtsp://192.168.1.64:554/Streaming/Channels/101?transportmode=unicast&profile=Test
# #rtsp://192.168.1.163:554/live/av0
# haar特征(霍尔),使用adaboost级联分类器
# 安装ffmpeg可以直接播放视频
video_capture = cv2.VideoCapture(0)
print(video_capture.isOpened())
# 使用haa特征检测人脸
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    # 读出视频帧，ret表示成功或失败
    ret, frame = video_capture.read()
    if ret:
        # 得到系统时钟时间
        t1 = cv2.getTickCount()
        # 对每一帧检测人脸
        faces = detect(frame, face_cascade)
        t2 = cv2.getTickCount()

        # 将检测到的人脸画上正方形标记
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 2)

        cv2.imshow('imshow', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
