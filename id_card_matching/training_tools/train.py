import os
import datetime
import numpy as np
import tensorflow as tf

# from model import get_siamese_model
from dl.fpn_model import get_siamese_model
from dl.losses import triplet_loss
from dl.metrics import triplet_accuracy

NUM_EPOCHS = 1000
BATCH_SIZE = 16

TRAIN_STEPS = 500
VAL_STEPS = 100
LEARNING_RATE = 0.0005
MIXED_PRECISION = False
DATA_DIR = "./lfw/lfw/"  # "./Data/"
NUM_FOLDERS = 1680  # 10575
DISTRIBUTE = False

import logging

logging.getLogger("tensorflow").disabled = True

if MIXED_PRECISION:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)

image_feature_description = {
    "anchor": tf.io.FixedLenFeature([], tf.string),
    "positive": tf.io.FixedLenFeature([], tf.string),
    "negative": tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


dataset = tf.data.TFRecordDataset("./triplet_data.tfrecords")
dataset = dataset.map(
    _parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
dataset = dataset.map(
    lambda x: (x["anchor"], x["positive"], x["negative"]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)


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


dataset = dataset.map(_process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

anchors = dataset.map(
    lambda x, y, z: x, num_parallel_calls=tf.data.experimental.AUTOTUNE
).batch(BATCH_SIZE)
positives = dataset.map(
    lambda x, y, z: y, num_parallel_calls=tf.data.experimental.AUTOTUNE
).batch(BATCH_SIZE)
negatives = dataset.map(
    lambda x, y, z: z, num_parallel_calls=tf.data.experimental.AUTOTUNE
).batch(BATCH_SIZE)

dataset = tf.data.Dataset.zip((anchors, positives, negatives))
train_dataset = dataset.map(
    lambda x, y, z: ((x, y, z), np.empty(0)),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).repeat()



val_dataset = tf.data.TFRecordDataset("./triplet_data_test.tfrecords")
val_dataset = val_dataset.map(
    _parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
val_dataset = val_dataset.map(
    lambda x: (x["anchor"], x["positive"], x["negative"]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)

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

train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


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

# model.load_weights(".\\logs\\cropped\\siamese.h5")

if __name__ == "__main__":
    log_dir = ".\\dl\\logs\\" + "cropped"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + "\\siamese.h5", verbose=1)

    # train_gen = get_triplet_data(BATCH_SIZE)
    # val_gen = get_triplet_data(BATCH_SIZE)
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VAL_STEPS,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint],
        workers=1,
        use_multiprocessing=False,
    )
