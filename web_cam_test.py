import cv2
import numpy as np

import predict
import id_database
import face_crop

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 640)  # set Height
driver = id_database.DataDriver()
cropper = face_crop.FaceCrop()

driver.add_person("Rohil", 1, cv2.imread("./ids/test_rohil.jpg"))
driver.add_person("Tanish", 2, cv2.imread("./ids/test_tanish.jpg"))
driver.add_person("Ayush", 3, cv2.imread("./ids/test_ayush.jpg"))
driver.add_person("Sai", 4, cv2.imread("./ids/test_sai.jpg"))
driver.add_person("Megna", 5, cv2.imread("./ids/test_megna.jpg"))
driver.add_person("Jayani", 6, cv2.imread("./ids/test_jayani.jpg"))

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip camera vertically

        id, name = driver.lookup_person(frame)
        # id, name = driver.lookup_person(cropper.crop(frame))
        cv2.imshow("web cam", frame)
        print(name)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()
