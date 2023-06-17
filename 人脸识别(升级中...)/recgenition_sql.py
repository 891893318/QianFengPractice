import cv2
import mysql.connector
import numpy as np
from PIL import ImageFont, ImageDraw, Image

connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Jiang20011010...',
    db='data_recognition'
)

# 加载识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner_sql/trainner_sql.yml')

# 加载分类器
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 开摄像头
cam = cv2.VideoCapture(0)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# 加载中文字体
font_path = "中文字体/STXINGKA.TTF"  # 字体文件路径
font_size = 24  # 字体大小

# 创建绘图对象
font = ImageFont.truetype(font_path, font_size)

names = []
myCursor_name = connection.cursor()
query = "SELECT s_name FROM member_message "
myCursor_name.execute(query)
# 读取结果集并保存到列表中
for (name,) in myCursor_name:
    names.append(name)
print(names)


while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        img_id, confidence = recognizer.predict(roi_gray)

        if confidence < 60:
            print(img_id)
            img_id = names[img_id]
            print(img_id)
            confidence = "{0}%".format(round(100 - confidence))

            myCursor = connection.cursor()
            sql = "SELECT * FROM member_message WHERE s_name = %s"
            val = (img_id,)
            myCursor.execute(sql, val)
            myResult = myCursor.fetchone()

            name = myResult[1]
            gender = myResult[2]
            student_id = myResult[3]
            age = myResult[4]
            college = myResult[5]
            text = f"姓名: {name}\n性别: {gender}\n学号: {student_id}\n年龄: {age}\n学院: {college}"
        else:
            img_id = "未知"
            confidence = "{0}%".format(round(100 - confidence))
            text = "陌生人，报警！！"

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        text_bbox = draw.textbbox((x, y + h), text, font=font)
        draw.rectangle(text_bbox, fill=(99, 128, 150, 128))
        draw.text(text_bbox[:2], text, font=font, fill=(255, 255, 255))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.putText(img, str(confidence), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow('face_recgnise', img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
