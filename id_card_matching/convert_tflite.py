import tensorflow as tf

from dl.model_simple import get_siamese_model
from training_tools.validation import val_dataset

def data_gen():
    for i in val_dataset.take(2):
        for j in i[0][0]:
            yield([j])

model = get_siamese_model(training=False)
# model.load_weights("D:\\id-card-matching\\logs\\cropped_small\\siamese.h5", by_name=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter=False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

converter.representative_dataset = data_gen

tflite_quant_model = converter.convert()

with open("model.tflite", "wb") as file:
    file.write(tflite_quant_model)