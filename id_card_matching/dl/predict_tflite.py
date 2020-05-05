import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

class Predictor:
    def __init__(self):
        self.threshold = 0.45
        self.tflite_model = tflite.Interpreter("./model_edgetpu.tflite", experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
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
        dis = tf.math.sqrt(
                tf.math.reduce_sum(tf.math.square(vec1 - vec2), axis=-1)
            )
        if dis < self.threshold:
            return True
        else:
            return False
    def load_and_preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.face_cropper.crop(image)
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(tf.reshape(
                tf.image.resize(image, (112, 112)),
            (1, 112, 112, 3),
        ))
        return image