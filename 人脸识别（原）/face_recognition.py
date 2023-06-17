import cv2
import mysql.connector
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 加载识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')

# 加载分类器
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 开摄像头
cam = cv2.VideoCapture(0)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# 加载中文字体
font_path = "../人脸识别(答辩项目)/中文字体/font.ttf"  # 字体文件路径
font_size = 24  # 字体大小

# 创建绘图对象
font = ImageFont.truetype(font_path, font_size)

names = ['黄代鑫','蒋洪波']

# 连接数据库
connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Jiang20011010...',
        db='data_recognition'
    )

while True:
    # 人脸识别(答辩项目)
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    # 遍历检测到的每张人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # 进行人脸识别
        img_id, confidence = recognizer.predict(roi_gray)
        print(img_id, confidence)

        if confidence < 50:  # 越小，可能性越高
            img_id = names[img_id - 1]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            img_id = "未知"
            confidence = "{0}%".format(round(100 - confidence))

        # 将名字绘制在图像上
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        text_bbox = draw.textbbox((x, y + h), img_id, font=font)
        draw.text(text_bbox[:2], img_id, font=font, fill=(0, 255, 0))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.putText(img, str(confidence), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow('im', img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
