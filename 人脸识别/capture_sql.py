import cv2
import mysql.connector
import base64
import os

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
sampleNum = 0
Id = input('输入人脸ID: ')
print('\n 正在初始化人脸采集，请注视摄像头 ...')

cam = cv2.VideoCapture(0)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Jiang20011010...',
    db='data_recognition'
)

# Create the 'dataSet' folder if it doesn't exist
if not os.path.exists('dataSet'):
    os.makedirs('dataSet')

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

                # Save the image data in the 'dataSet' folder
                image_path = f"dataSet/{Id}_{sampleNum}.jpg"
                cv2.imwrite(image_path, gray[y:y + h, x:x + w])

                cv2.imshow('frame', img)

                # Insert the image path into the database
                sql = "INSERT INTO images_table (image_path, image_id) VALUES (%s, %s)"
                cursor.execute(sql, (image_path, Id))
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
