import cv2
import mysql.connector
import numpy as np
import base64


# LBPH模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def get_images_and_labels():
    """
    将数据库中的二进制图像信息转换为训练数据和标签
    :return: face_samples: 人脸数据列表, ids: 标签列表
    """
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Jiang20011010...',
        db='data_recognition',
    )

    try:
        with connection.cursor() as cursor:
            # 执行 SQL 查询
            sql = "SELECT image_path, image_id FROM images_table"
            cursor.execute(sql)
            result = cursor.fetchall()

            face_samples = []  # 存放人脸数据
            ids = []  # 存放标签

            for row in result:
                image_data = row[0]
                image_id = row[1]

                # 将二进制图像数据转换为numpy数组
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                # print(nparr)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                # 检测图片中是否存在人脸
                faces = detector.detectMultiScale(image_np)

                for (x, y, w, h) in faces:
                    # 生成训练数据
                    face_samples.append(image_np[y:y + h, x:x + w])
                    # 数据标记
                    ids.append(image_id)

        return face_samples, ids

    finally:
        connection.close()

get_images_and_labels()
faces, Ids = get_images_and_labels()
Ids = np.array(Ids, dtype=np.int32)

recognizer.train(faces, np.array(Ids))
recognizer.save("trainner_sql/trainner_sql.yml")
