import cv2
import numpy as np

import predict
import id_database


cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 640)  # set Height
driver = id_database.DataDriver()
driver.add_person("rohil", 1111, cv2.imread("./ids/test_rohil.jpg"))
driver.add_person("sai", 2222, cv2.imread("./ids/test_sai.jpg"))
driver.add_person("ayush", 3333, cv2.imread("./ids/test_ayush.jpg"))


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip camera vertically

    name, id = driver.lookup_person(frame)
    cv2.imshow("web cam", frame)
    print(name)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()