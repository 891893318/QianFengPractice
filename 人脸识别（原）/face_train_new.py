import cv2
import os
import numpy as np


from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

# LBPH模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def get_images_and_labels(path):
    """
    将图片变成训练数据和标签
    :param path:
    :return:
    """
    # 读取所有的文件名
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []  # 存人脸数据
    ids = []  # 存标签

    print(image_paths)  # 测试

    # 便利每一个人脸文件
    for image_path in image_paths:
        image_np = cv2.imread(image_path, 0)
        print((image_path))
        # print(image_np)  #测试
        if os.path.split(image_path)[-1].split(".")[-1] != "jpg":
            continue

        # 取出人脸label，是谁的脸
        image_id = int(os.path.split(image_path)[-1].split(".")[1])

        # 检测图片中是否存在人脸
        faces = detector.detectMultiScale(image_np)

        for (x, y, w, h) in faces:
            # 生成训练数据
            face_samples.append(image_np[y:y + h, x:x + w])
            # 数据标记
            ids.append(image_id)

    return face_samples, ids


faces, Ids = get_images_and_labels("dataSet")
print(type(Ids))
recognizer.train(faces, np.array(Ids))
recognizer.save("trainner/trainner.yml")

# lr = LinearRegression()
# joblib.dump(lr,"./test.yml")
# joblib.load("")
