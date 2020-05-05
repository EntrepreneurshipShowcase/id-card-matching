import tensorflow as tf

from dl.fpn_model import get_siamese_model
from training_tools.validation import val_dataset
import os

def data_gen():
    for i in os.listdir("../temp_images"):
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - 127.5)/127.5
        yield [[img]]

model = get_siamese_model(training=False)
# model = tf.keras.applications.ResNet50(include_top=False, input_shape=(96, 96, 3))
model.load_weights("../siamese.h5", by_name=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.experimental_new_converter=False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

converter.representative_dataset = data_gen

tflite_quant_model = converter.convert()

with open("model.tflite", "wb") as file:
    file.write(tflite_quant_model)
