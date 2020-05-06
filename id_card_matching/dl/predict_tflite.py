import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from dl.face_crop import FaceCrop

class Predictor:
    def __init__(self):
        self.threshold = 0.55
        self.tflite_model = tflite.Interpreter("../model_edgetpu.tflite", experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
        self.tflite_model.allocate_tensors()
        self.face_cropper = FaceCrop()
        self.input_details = self.tflite_model.get_input_details()
        self.output_details = self.tflite_model.get_output_details()
    def l2_norm(self, x, axis=1):
        """l2 norm"""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        output = x / norm

        return output
    def get_vec(self, image):
        # import ipdb; ipdb.set_trace()
        image = self.load_and_preprocess_image(image)
        #cv2.imwrite(image, "./test.jpg")
        print("Generating Vec")
        self.tflite_model.set_tensor(self.input_details[0]['index'], image)
        self.tflite_model.invoke()
        vec = self.tflite_model.get_tensor(self.output_details[0]["index"])
        vec = self.l2_norm(vec)
        if len(vec) == 1:
            return vec[0]
        elif len(vec) > 1:
            print("caution, multiple faces detected, choosing one at random")
            return vec[0]
        else:
            print("No face detected, return null")
            return np.zeros((128, ))
    def is_same_face(self, vec1, vec2, debug=True):
        dis = cosine(vec1, vec2)
        if dis < self.threshold:
            return True
        else:
            return False
    def load_and_preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.face_cropper.crop(image)
        image = tf.image.resize(image, (224, 224))
        x_temp = np.copy(image)
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] -= 91.4953
        x_temp[..., 1] -= 103.8827
        x_temp[..., 2] -= 131.0912
        x_temp = np.reshape(x_temp, (1, 224, 224, 3))
        return x_temp
