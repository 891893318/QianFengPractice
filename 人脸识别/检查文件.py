import cv2
import mysql.connector
import base64
import numpy as np

# Connect to the database
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Jiang20011010...',
    db='data_recognition'
)

try:
    with connection.cursor() as cursor:
        # Prompt user to enter the image ID
        image_id = input("Enter the image ID: ")

        # Retrieve image data from the database
        sql = "SELECT image_path FROM images_table WHERE image_id = %s"
        cursor.execute(sql, (image_id,))
        result = cursor.fetchone()

        if result is not None:
            image_data = result[0]

            # Decode Base64 string back to image data
            try:
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Display the image
                cv2.imshow("Image", img)
                cv2.waitKey(0)
            except base64.binascii.Error:
                print("Invalid base64-encoded image data.")
        else:
            print("Image not found.")

        # Fetch any remaining results
        cursor.fetchall()

finally:
    connection.close()
    cv2.destroyAllWindows()
