import cv2

#结构光
#tof
#单摄像头
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
sampleNum = 0
Id = input('输入人脸ID: ')
print('\n 正在初始化人脸采集，请注视摄像头 ...')

cam = cv2.VideoCapture(0)
minW = 0.1 * cam.get(3) #宽
minH = 0.1 * cam.get(4) #高


while True:
    #读取一帧
    ret, img = cam.read()
    #将彩色图转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(int(minW), int(minH))
    )
    for (x, y, w, h) in faces:
        #opencv绘制正方形需要左上角坐标和右下角坐标,人脸检测检测出得是左上角坐标和宽高
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #增加样本计数
        sampleNum = sampleNum + 1
        # 将样本拷贝到指定文件夹
        #注意:opencv的宽高和实际是相反的
        cv2.imwrite("dataSet/User." + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])  #

        cv2.imshow('frame', img)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        print(sampleNum)
        break

    elif sampleNum >= 100:
        print(sampleNum)
        break

cam.release()
cv2.destroyAllWindows()

#暗光环境拍摄一张照片,opencv进行降噪（可选）
#采集至少三个样本进行训练
#数据分析常见面试题 https://github.com/yoghurtjia/-python-BAT-(重点)


