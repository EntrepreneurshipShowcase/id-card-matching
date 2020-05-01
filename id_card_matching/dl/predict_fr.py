import face_recognition
import cv2

class Predictor:
    def __init__(self):
        self.threshold = 70
    def get_vec(self, image):
        image = self.load_and_preprocess_image(image)
        # cv2.imwrite(image[0], "./test.jpg")
        # import ipdb; ipdb.set_trace()
        vec = face_recognition.face_encodings(image)[0]
        return vec
    def is_same_face(self, vec1, vec2, debug=True):
        dis = face_recognition.face_distance(vec1, vec2)
        print(dis)
        if dis < self.threshold:
            return True
        else:
            return False
    def load_and_preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image