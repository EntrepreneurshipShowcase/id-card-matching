import face_recognition
import cv2
import numpy as np

class Predictor:
    def __init__(self):
        self.threshold = 0.45
    def get_vec(self, image):
        # import ipdb; ipdb.set_trace()
        image = self.load_and_preprocess_image(image)
        #cv2.imwrite(image, "./test.jpg")
        print("Generating Vec")
        vec = face_recognition.face_encodings(image)
        if len(vec) == 1:
            return vec[0]
        elif len(vec) > 1:
            print("caution, multiple faces detected, choosing one at random")
            return vec[0]
        else:
            print("No face detected, return null")
            return np.zeros((128, ))
    def is_same_face(self, vec1, vec2, debug=True):
        dis = face_recognition.face_distance([vec1], vec2)[0]
        print(dis)
        if dis < self.threshold:
            return True
        else:
            return False
    def load_and_preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image