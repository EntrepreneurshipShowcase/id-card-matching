import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

from dl.losses import triplet_loss
from dl.metrics import triplet_accuracy

EMBEDDING_LAYER_DIM = 128

class Embedding(layers.Layer):
    def __init__(self):
        super(Embedding, self).__init__()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.conv1 = layers.Conv2D(EMBEDDING_LAYER_DIM*2, (2, 2))
        # self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))
        # self.bn2 = layers.BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = layers.Activation("relu")(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = layers.Activation("relu")(x)
        x = self.global_pool(x)
        x = layers.Activation("linear", dtype=tf.float32)(x)
        return x

class TripletLoss(layers.Layer):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, anchor, positive, negative):
        self.add_loss(triplet_loss(anchor, positive, negative, self.margin))
        return (anchor, positive, negative)

    def get_config(self):
        return {"margin": self.margin}


class TripletAccuracy(layers.Layer):
    def __init__(self, margin):
        super(TripletAccuracy, self).__init__()
        self.margin = margin

    def call(self, anchor, positive, negative):
        self.add_metric(
            triplet_accuracy(anchor, positive, negative, self.margin),
            name="accuracy_" + str(self.margin),
            aggregation="mean",
        )
        return (anchor, positive, negative)

    def get_config(self):
        return {"margin": self.margin}

def get_siamese_model(training=True):
    inp_shape = (96, 96, 3)
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=inp_shape, weights="imagenet")
    backbone = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    embedding = Embedding()
    if training:
        input_a = layers.Input(inp_shape, name="anchor")
        input_p = layers.Input(inp_shape, name="positive")
        input_n = layers.Input(inp_shape, name="negative")

        b_a = backbone(input_a)
        b_p = backbone(input_p)
        b_n = backbone(input_n)

        a = embedding(b_a)
        p = embedding(b_p)
        n = embedding(b_n)

        a, p, n = TripletLoss(10)(a, p, n)
        a, p, n = TripletAccuracy(10)(a, p, n)
        out = layers.Concatenate()([a, p, n])
        model = tf.keras.models.Model(
            inputs=[input_a, input_p, input_n], outputs=out
        )
        return model
    else:
        input_x = layers.Input(inp_shape)
        x = backbone(input_x)
        x = embedding(x)
        model = tf.keras.models.Model(inputs=input_x, outputs=x)
        return model
