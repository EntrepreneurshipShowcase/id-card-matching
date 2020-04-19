import os
import datetime
import numpy as np
import tensorflow as tf

from model_large import get_siamese_model
from losses import triplet_loss
from metrics import triplet_accuracy

BATCH_SIZE = 32
DISTRIBUTE = False
LEARNING_RATE = 0.0001

import logging

logging.getLogger("tensorflow").disabled = True


image_feature_description = {
    "anchor": tf.io.FixedLenFeature([], tf.string),
    "positive": tf.io.FixedLenFeature([], tf.string),
    "negative": tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


val_dataset = tf.data.TFRecordDataset("./triplet_data_test.tfrecords")
val_dataset = val_dataset.map(
    _parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
val_dataset = val_dataset.map(
    lambda x: (x["anchor"], x["positive"], x["negative"]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
import ipdb; ipdb.set_trace()


# def _process_image(anchor, positive, negative):
#     anchor_img = tf.reshape(
#         tf.image.resize(
#             tf.reshape(
#                 tf.keras.applications.inception_resnet_v2.preprocess_input(
#                     tf.io.decode_image(anchor, dtype=tf.float32)
#                 ),
#                 (250, 250, 3),
#             ),
#             (224, 224),
#         ),
#         (224, 224, 3),
#     )
#     positive_img = tf.reshape(
#         tf.image.resize(
#             tf.reshape(
#                 tf.keras.applications.inception_resnet_v2.preprocess_input(
#                     tf.io.decode_image(positive, dtype=tf.float32)
#                 ),
#                 (250, 250, 3),
#             ),
#             (224, 224),
#         ),
#         (224, 224, 3),
#     )
#     negative_img = tf.reshape(
#         tf.image.resize(
#             tf.reshape(
#                 tf.keras.applications.inception_resnet_v2.preprocess_input(
#                     tf.io.decode_image(negative, dtype=tf.float32)
#                 ),
#                 (250, 250, 3),
#             ),
#             (224, 224),
#         ),
#         (224, 224, 3),
#     )
#     return anchor_img, positive_img, negative_img
def _process_image(anchor, positive, negative):
    anchor_img = tf.keras.applications.inception_resnet_v2.preprocess_input(tf.reshape(
        tf.image.resize(
            tf.reshape(
                    tf.io.decode_image(anchor, dtype=tf.float32), (64, 64, 3),
            ),
            (96, 96),
        ),
        (96, 96, 3),
    ))
    positive_img = tf.keras.applications.inception_resnet_v2.preprocess_input(tf.reshape(
        tf.image.resize(
            tf.reshape(
                    tf.io.decode_image(positive, dtype=tf.float32), (64, 64, 3),
            ),
            (96, 96),
        ),
        (96, 96, 3),
    ))
    negative_img = tf.keras.applications.inception_resnet_v2.preprocess_input(tf.reshape(
        tf.image.resize(
            tf.reshape(
                    tf.io.decode_image(negative, dtype=tf.float32), (64, 64, 3),
            ),
            (96, 96),
        ),
        (96, 96, 3),
    ))
    return anchor_img, positive_img, negative_img

val_dataset = val_dataset.map(
    _process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
)

anchors = val_dataset.map(
    lambda x, y, z: x, num_parallel_calls=tf.data.experimental.AUTOTUNE
).batch(BATCH_SIZE)
positives = val_dataset.map(
    lambda x, y, z: y, num_parallel_calls=tf.data.experimental.AUTOTUNE
).batch(BATCH_SIZE)
negatives = val_dataset.map(
    lambda x, y, z: z, num_parallel_calls=tf.data.experimental.AUTOTUNE
).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.zip((anchors, positives, negatives))
val_dataset = val_dataset.map(
    lambda x, y, z: ((x, y, z), np.empty(0)),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).repeat()


def scheduler(epoch):
    if epoch <= 200:
        return 2e-4
    else:
        return 2e-4 * 10 ** (-3 * (epoch - 200) / 200)


if DISTRIBUTE:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_siamese_model()
        optimizer = tf.keras.optimizers.Adam(
            lr=LEARNING_RATE, beta_1=0.99, beta_2=0.999, epsilon=1e-3
        )
        model.compile(optimizer)
else:
    model = get_siamese_model()
    optimizer = tf.keras.optimizers.Adam(
        lr=LEARNING_RATE, beta_1=0.99, beta_2=0.999, epsilon=1e-3
    )
    model.compile(optimizer)

model.load_weights(".\\logs\\training_large_margin\\siamese.h5")

if __name__ == "__main__":
    model.evaluate(val_dataset, steps=200)
