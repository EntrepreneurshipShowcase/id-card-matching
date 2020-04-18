import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import inception_resnet_v2
import os
import datetime

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)


def get_triplet_data(total, batch_size):
    i = 0
    anchor_list, positive_list, negative_list = (
        np.zeros((batch_size, 250, 250, 3)),
        np.zeros((batch_size, 250, 250, 3)),
        np.zeros((batch_size, 250, 250, 3)),
    )
    while True:  # i < total:
        for idx in range(batch_size):
            face_id = np.random.randint(0, 10575)
            negative_id = np.random.randint(0, 10575)
            while negative_id == face_id:
                negative_id = np.random.randint(0, 10575)
            anchor = np.random.choice(os.listdir("./Data/{:05d}".format(face_id)))
            positive = np.random.choice(os.listdir("./Data/{:05d}".format(face_id)))
            while positive == anchor:
                positive = np.random.choice(os.listdir("./Data/{:05d}".format(face_id)))
            negative = np.random.choice(os.listdir("./Data/{:05d}".format(negative_id)))
            negative_path = "{:05d}".format(negative_id)
            face_path = "{:05d}".format(face_id)
            negative_image = inception_resnet_v2.preprocess_input(
                tf.io.decode_image(
                    tf.io.read_file(f"./Data/{negative_path}/{negative}"),
                    dtype=tf.float32,
                )
            )
            anchor_image = inception_resnet_v2.preprocess_input(
                tf.io.decode_image(
                    tf.io.read_file(f"./Data/{face_path}/{anchor}"), dtype=tf.float32
                )
            )
            positive_image = inception_resnet_v2.preprocess_input(
                tf.io.decode_image(
                    tf.io.read_file(f"./Data/{face_path}/{positive}"), dtype=tf.float32
                )
            )
            anchor_list[idx] = anchor_image
            positive_list[idx] = positive_image
            negative_list[idx] = negative_image
        yield (anchor_list, positive_list, negative_list), np.empty(1)
        i += 1


def triplet_loss(y_true, y_pred, cosine=True, alpha=0.4):
    embedding_size = y_pred.shape[-1] // 3
    ind = int(embedding_size * 2)
    a_pred = y_pred[:, :embedding_size]
    p_pred = y_pred[:, embedding_size:ind]
    n_pred = y_pred[:, ind:]
    if cosine:
        positive_distance = 1 - tf.math.reduce_sum((a_pred * p_pred), axis=-1)
        negative_distance = 1 - tf.math.reduce_sum((a_pred * n_pred), axis=-1)
    else:
        positive_distance = tf.math.sqrt(
            tf.math.reduce_sum(tf.math.square(a_pred - p_pred), axis=-1)
        )
        negative_distance = tf.math.sqrt(
            tf.math.reduce_sum(tf.math.square(a_pred - n_pred), axis=-1)
        )
    loss = tf.math.maximum(0.0, positive_distance - negative_distance + alpha)
    return loss


inception_resnet = inception_resnet_v2.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=(250, 250, 3)
)
output = layers.GlobalAveragePooling2D()(inception_resnet.output)
base_model = tf.keras.models.Model(inception_resnet.input, output)


def embedder(conv_feat_size):
    input_x = layers.Input((conv_feat_size,), name="input")
    normalize = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=-1), name="normalize"
    )
    x = layers.Dense(512)(input_x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(128)(x)
    x = normalize(x)
    model = tf.keras.models.Model(input_x, x)
    return model


def get_siamese_model(base_model):
    inp_shape = (250, 250, 3)
    conv_feat_size = base_model.output.shape[-1]

    input_a = layers.Input(inp_shape, name="anchor")
    input_p = layers.Input(inp_shape, name="positive")
    input_n = layers.Input(inp_shape, name="negative")

    emb_model = embedder(conv_feat_size)
    output_a = emb_model(base_model(input_a))
    output_p = emb_model(base_model(input_p))
    output_n = emb_model(base_model(input_n))

    merged_vector = layers.Concatenate(axis=-1)([output_a, output_p, output_n])
    merged_vector = layers.Activation("linear", dtype="float32")(merged_vector)
    model = tf.keras.models.Model(
        inputs=[input_a, input_p, input_n], outputs=merged_vector
    )

    return model


model = get_siamese_model(base_model)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer, loss=triplet_loss)

log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    log_dir + "\\siamese.h5", monitor="val_loss", verbose=1, save_best_only=True
)

train_gen = get_triplet_data(1000, 16)
val_gen = get_triplet_data(100, 16)

# train_dataset = tf.data.Dataset.from_generator(lambda:get_triplet_data(1000), tf.float32, output_shapes=(3, 250, 250, 3))
# train_dataset = train_dataset.batch(32, drop_remainder = True)

# val_dataset = tf.data.Dataset.from_generator(lambda:get_triplet_data(100), tf.float32, output_shapes=(3, 250, 250, 3))
# val_dataset = train_dataset.batch(32, drop_remainder = True)

# model.fit(train_dataset, validation_data=val_dataset, epochs=1, callbacks=[checkpoint])


model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=100,
    validation_steps=10,
    epochs=100,
    callbacks=[checkpoint, tensorboard_callback],
)
# for _ in range(2):
#     for train_x, _ in train_gen:
#         import pdb; pdb.set_trace()
#         with tf.GradientTape() as tape:
#             out = model(train_x)
#             loss_val = triplet_loss([], out)
#         grads = tape.gradient(loss_val, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
