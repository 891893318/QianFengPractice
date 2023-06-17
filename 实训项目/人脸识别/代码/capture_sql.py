import cv2
import mysql.connector
import base64

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
sampleNum = 0

# Input student information
Id = input('输入人脸ID: ')
name = input('输入学生姓名: ')
gender = input('输入学生性别: ')
student_id = input('输入学生学号: ')
age = input('输入学生年龄: ')
college = input('输入学生学院: ')

print('\n 正在初始化人脸采集，请注视摄像头 ...')

cam = cv2.VideoCapture(0)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Jiang20011010...',
    db='data_recognition_picture'
)

try:
    with connection.cursor() as cursor:
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(int(minW), int(minH))
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1

                # 将捕获的人脸图像数据与学生信息保存到数据库中
                _, buffer = cv2.imencode('.jpg', gray[y:y + h, x:x + w])
                image_data = base64.b64encode(buffer).decode('utf-8')

                cv2.imshow('frame', img)

                # 将捕获的人脸图像数据与学生信息保存到数据库中
                sql = "INSERT INTO images_table (image_path, image_id, s_name, sex, student_id, age, college) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (image_data, Id, name, gender, student_id, age, college))
                connection.commit()

            if cv2.waitKey(2) & 0xFF == ord('q'):
                print(sampleNum)
                break

            elif sampleNum >= 100:
                print(sampleNum)
                break

finally:
    connection.close()
    cam.release()
    cv2.destroyAllWindows()
