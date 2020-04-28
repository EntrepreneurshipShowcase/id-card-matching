import tensorflow as tf
import os
import cv2

from dl.fpn_model import get_siamese_model


class Predictor:
    def __init__(self):
        self.model = get_siamese_model(training=False)
        self.model.load_weights(
            "./dl/siamese.h5", by_name=True
        )
        #TODO Determine optimal threshold
        self.threshold = 95
    #TODO Fix input processing
    def load_and_preprocess_files(self, id_image_file, test_image_file):
        id_image = tf.keras.applications.inception_resnet_v2.preprocess_input(tf.reshape(
            tf.image.resize(
                        tf.io.decode_image(
                            tf.io.read_file(id_image_file), dtype=tf.float32
                        ),
                (96, 96),
            ),
            (1, 96, 96, 3),
        ))
        test_image = tf.reshape(
            tf.image.resize(
                tf.reshape(
                    tf.keras.applications.inception_resnet_v2.preprocess_input(
                        tf.io.decode_image(
                            tf.io.read_file(test_image_file), dtype=tf.float32
                        )
                    ),
                    (250, 250, 3),
                ),
                (224, 224),
            ),
            (1, 224, 224, 3),
        )
        return id_image, test_image

    def load_and_preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(tf.reshape(
                tf.image.resize(image, (224, 224)),
            (1, 224, 224, 3),
        ))
        return image

    def get_distance(self, id_vec, test_vec, cosine=False):
        if cosine:
            distance = 1 - tf.math.reduce_sum((id_vec * test_vec), axis=-1)
        else:
            distance = tf.math.sqrt(
                tf.math.reduce_sum(tf.math.square(id_vec - test_vec), axis=-1)
            )
        return distance

    def predict_on_file(self, id_image_file, test_image_file):
        id_image, test_image = self.load_and_preprocess_files(
            id_image_file, test_image_file
        )
        id_vec = self.model.predict(id_image)
        test_vec = self.model.predict(test_image)
        print(self.get_distance(id_vec, test_vec))
        if self.get_distance(id_vec, test_vec) < self.threshold:
            return True
        else:
            return False

    def get_vec(self, image):
        image = self.load_and_preprocess_image(image)
        # cv2.imwrite(image[0], "./test.jpg")
        # import ipdb; ipdb.set_trace()
        vec = self.model.predict(image)[0]
        return vec
    def is_same_face(self, vec1, vec2, debug=True):
        dis = self.get_distance(vec1, vec2)
        print(dis)
        if dis < self.threshold:
            return True
        else:
            return False
    def vec_distance(self, id_image, test_image):
        id_vec = self.get_vec(id_image)
        test_vec = self.get_vec(test_image)
        return self.get_distance(id_vec, test_vec)
        # if self.get_distance(id_vec, test_vec) < self.threshold:
        #     return True
        # else:
        #     return False


# x = Predictor()
