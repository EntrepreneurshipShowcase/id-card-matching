import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from losses import triplet_loss
from metrics import triplet_accuracy
from tensorflow.keras.applications import resnet_v2

EMBEDDING_LAYER_DIM = 256


class BaseEmbedding(layers.Layer):
    def __init__(self):
        super(BaseEmbedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1), activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.global_pool(x)
        x = layers.Activation("linear", dtype=tf.float32)(x)
        return x


class Shift1Embedding(layers.Layer):
    def __init__(self):
        super(Shift1Embedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(512, (3, 3), activation="relu")
        self.bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1), activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = layers.Activation("linear", dtype=tf.float32)(x)
        return x


class Shift2Embedding(layers.Layer):
    def __init__(self):
        super(Shift2Embedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(256, (3, 3), (2, 2), activation="relu")
        self.bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1), activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
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
    inp_shape = (299, 299, 3)
    base_model = resnet_v2.ResNet50V2(include_top=False, weights="imagenet")
    base_model.trainable=False

    if training:
        input_a = layers.Input(inp_shape, name="anchor")
        input_p = layers.Input(inp_shape, name="positive")
        input_n = layers.Input(inp_shape, name="negative")

        # base = tf.keras.Model(
        #     inputs=base_model.input,
        #     outputs=(
        #         base_model.output,
        #         base_model.get_layer("block_11_add").output,
        #         base_model.get_layer("block_7_add").output,
        #     ),
        # )
        base = tf.keras.Model(inputs=base_model.input, outputs=(base_model.output, base_model.get_layer("conv4_block6_out").output, base_model.get_layer("conv3_block4_out").output))

        base_embedding = BaseEmbedding()
        shift1_embedding = Shift1Embedding()
        shift2_embedding = Shift2Embedding()
        base_a, shifted1_a, shifted2_a = base(input_a)
        base_a = base_embedding(base_a)

        base_p, shifted1_p, shifted2_p = base(input_p)
        base_p = base_embedding(base_p)

        base_n, shifted1_n, shifted2_n = base(input_n)
        base_n = base_embedding(base_n)

        base_a, base_p, base_n = TripletLoss(margin=4)(base_a, base_p, base_n)
        # base_a, base_p, base_n = TripletAccuracy(margin=55)(base_a, base_p, base_n)
        # TripletLoss(margin=4)(base_a, base_p, base_n)

        shifted1_a = shift1_embedding(shifted1_a)
        shifted1_a = layers.Add()([shifted1_a, base_a])

        shifted1_p = shift1_embedding(shifted1_p)
        shifted1_p = layers.Add()([shifted1_p, base_p])

        shifted1_n = shift1_embedding(shifted1_n)
        shifted1_n = layers.Add()([shifted1_n, base_n])

        shifted1_a, shifted1_p, shifted1_n = TripletLoss(margin=7)(
            shifted1_a, shifted1_p, shifted1_n
        )
        # shifted1_a, shifted1_p, shifted1_n = TripletAccuracy(margin=10)(
        #     shifted1_a, shifted1_p, shifted1_n
        # )
        # TripletLoss(margin=7)(shifted1_a, shifted1_p, shifted1_n)

        shifted2_a = shift2_embedding(shifted2_a)
        shifted2_a = layers.Add()([shifted2_a, shifted1_a])

        shifted2_p = shift2_embedding(shifted2_p)
        shifted2_p = layers.Add()([shifted2_p, shifted1_p])

        shifted2_n = shift2_embedding(shifted2_n)
        shifted2_n = layers.Add()([shifted2_n, shifted1_n])

        shifted2_a, shifted2_p, shifted2_n = TripletLoss(margin=10)(
            shifted2_a, shifted2_p, shifted2_n
        )
        # shifted2_a, shifted2_p, shifted2_n = TripletAccuracy(margin=10)(
        #     shifted2_a, shifted2_p, shifted2_n
        # )
        # TripletLoss(margin=10)(shifted2_a, shifted2_p, shifted2_n)

        shifted2 = layers.Concatenate()([shifted2_a, shifted2_p, shifted2_n])

        model = tf.keras.models.Model(
            inputs=[input_a, input_p, input_n], outputs=shifted2
        )
    else:
        input_x = layers.Input(inp_shape, name="input")

        base = tf.keras.models.Model(base_model.input, (base_model.output, base_model.get_layer("conv4_block6_out").output, base_model.get_layer("conv3_block4_out").output))
        # base = tf.keras.Model(
        #     inputs=base_model.input,
        #     outputs=(
        #         base_model.output,
        #         base_model.get_layer("block_11_add").output,
        #         base_model.get_layer("block_7_add").output,
        #     ),
        # )

        base_embedding = BaseEmbedding()
        shift1_embedding = Shift1Embedding()
        shift2_embedding = Shift2Embedding()

        base_x, shifted1_x, shifted2_x = base(input_x)

        base_x = base_embedding(base_x)

        shifted1_x = shift1_embedding(shifted1_x)
        shifted1_x = layers.Add()([shifted1_x, base_x])

        shifted2_x = shift2_embedding(shifted2_x)
        shifted2_x = layers.Add()([shifted2_x, shifted1_x])

        model = tf.keras.models.Model(inputs=input_x, outputs=shifted2_x)

    return model
