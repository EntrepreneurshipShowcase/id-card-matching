import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import mobilenet_v2

from dl.losses import triplet_loss
from dl.metrics import triplet_accuracy

EMBEDDING_LAYER_DIM = 512


class BaseEmbedding(layers.Layer):
    def __init__(self):
        super(BaseEmbedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.global_pool(x)
        x = layers.Activation("linear", dtype=tf.float32)(x)
        return x


class Shift1Embedding(layers.Layer):
    def __init__(self):
        super(Shift1Embedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(1024, (3, 3))
        self.bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))

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
        self.conv1 = layers.Conv2D(512, (3, 3), (2, 2))
        self.bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = layers.Activation("linear", dtype=tf.float32)(x)
        return x

class Shift3Embedding(layers.Layer):
    def __init__(self):
        super(Shift3Embedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(256, (7, 7), (3, 3))
        self.bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = layers.Activation("linear", dtype=tf.float32)(x)
        return x
class Shift4Embedding(layers.Layer):
    def __init__(self):
        super(Shift4Embedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(128, (11, 11), (4, 4))
        self.bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))

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
    inp_shape = (224, 224, 3)
    base_model = mobilenet_v2.MobileNetV2(include_top=False, input_shape=(224, 224, 3))

    if training:
        input_a = layers.Input(inp_shape, name="anchor")
        input_p = layers.Input(inp_shape, name="positive")
        input_n = layers.Input(inp_shape, name="negative")

        base = tf.keras.Model(
            inputs=base_model.input,
            outputs=(
                base_model.output,
                base_model.get_layer("block_14_add").output,
                base_model.get_layer("block_11_add").output,
                base_model.get_layer("block_7_add").output,
                base_model.get_layer("block_4_add").output,
            ),
        )
        # base = tf.keras.Model(inputs=base_model.input, outputs=(base_model.output, base_model.get_layer("conv4_block6_out").output, base_model.get_layer("conv3_block4_out").output))

        base_embedding = BaseEmbedding()
        shift1_embedding = Shift1Embedding()
        shift2_embedding = Shift2Embedding()
        shift3_embedding = Shift3Embedding()
        shift4_embedding = Shift4Embedding()

        base_a, shifted1_a, shifted2_a, shifted3_a, shifted4_a = base(input_a)
        base_a = base_embedding(base_a)

        base_p, shifted1_p, shifted2_p, shifted3_p, shifted4_p = base(input_p)
        base_p = base_embedding(base_p)

        base_n, shifted1_n, shifted2_n, shifted3_n, shifted4_n = base(input_n)
        base_n = base_embedding(base_n)

        base_a, base_p, base_n = TripletLoss(margin=4)(base_a, base_p, base_n)
        base_a, base_p, base_n = TripletAccuracy(margin=55)(base_a, base_p, base_n)
        # TripletLoss(margin=4)(base_a, base_p, base_n)

        shifted1_a = shift1_embedding(shifted1_a)
        shifted1_a = layers.Add()([shifted1_a, base_a])

        shifted1_p = shift1_embedding(shifted1_p)
        shifted1_p = layers.Add()([shifted1_p, base_p])

        shifted1_n = shift1_embedding(shifted1_n)
        shifted1_n = layers.Add()([shifted1_n, base_n])

        shifted1_a, shifted1_p, shifted1_n = TripletLoss(margin=10)(
            shifted1_a, shifted1_p, shifted1_n
        )
        shifted1_a, shifted1_p, shifted1_n = TripletAccuracy(margin=10)(
            shifted1_a, shifted1_p, shifted1_n
        )
        # TripletLoss(margin=7)(shifted1_a, shifted1_p, shifted1_n)

        shifted2_a = shift2_embedding(shifted2_a)
        shifted2_a = layers.Add()([shifted2_a, shifted1_a])

        shifted2_p = shift2_embedding(shifted2_p)
        shifted2_p = layers.Add()([shifted2_p, shifted1_p])

        shifted2_n = shift2_embedding(shifted2_n)
        shifted2_n = layers.Add()([shifted2_n, shifted1_n])
        shifted2_a, shifted2_p, shifted2_n = TripletLoss(margin=15)(
            shifted2_a, shifted2_p, shifted2_n
        )
        shifted2_a, shifted2_p, shifted2_n = TripletAccuracy(margin=15)(
            shifted2_a, shifted2_p, shifted2_n
        )

        shifted3_a = shift3_embedding(shifted3_a)
        shifted3_a = layers.Add()([shifted3_a, shifted2_a])

        shifted3_p = shift3_embedding(shifted3_p)
        shifted3_p = layers.Add()([shifted3_p, shifted2_p])

        shifted3_n = shift3_embedding(shifted3_n)
        shifted3_n = layers.Add()([shifted3_n, shifted2_n])

        shifted3_a, shifted3_p, shifted3_n = TripletLoss(margin=25)(
            shifted3_a, shifted3_p, shifted3_n
        )
        shifted3_a, shifted3_p, shifted3_n = TripletAccuracy(margin=25)(
            shifted3_a, shifted3_p, shifted3_n
        )
        shifted4_a = shift4_embedding(shifted4_a)
        shifted4_a = layers.Add()([shifted4_a, shifted4_a])

        shifted4_p = shift4_embedding(shifted4_p)
        shifted4_p = layers.Add()([shifted4_p, shifted4_p])

        shifted4_n = shift4_embedding(shifted4_n)
        shifted4_n = layers.Add()([shifted4_n, shifted4_n])

        shifted4_a, shifted4_p, shifted4_n = TripletLoss(margin=40)(
            shifted4_a, shifted4_p, shifted4_n
        )
        shifted4_a, shifted4_p, shifted4_n = TripletAccuracy(margin=200)(
            shifted4_a, shifted4_p, shifted4_n
        )
        # TripletLoss(margin=10)(shifted2_a, shifted2_p, shifted2_n)

        shifted4 = layers.Concatenate()([shifted4_a, shifted4_p, shifted4_n])

        model = tf.keras.models.Model(
            inputs=[input_a, input_p, input_n], outputs=shifted4
        )
    else:
        input_x = layers.Input(inp_shape, name="input")

        # base = tf.keras.models.Model(base_model.input, (base_model.output, base_model.get_layer("conv4_block6_out").output, base_model.get_layer("conv3_block4_out").output))
        base = tf.keras.Model(
            inputs=base_model.input,
            outputs=(
                base_model.output,
                base_model.get_layer("block_14_add").output,
                base_model.get_layer("block_11_add").output,
                base_model.get_layer("block_7_add").output,
                base_model.get_layer("block_4_add").output,
            ),
        )

        base_embedding = BaseEmbedding()
        shift1_embedding = Shift1Embedding()
        shift2_embedding = Shift2Embedding()
        shift3_embedding = Shift3Embedding()
        shift4_embedding = Shift4Embedding()

        base_x, shifted1_x, shifted2_x, shifted3_x, shifted4_x = base(input_x)

        base_x = base_embedding(base_x)

        shifted1_x = shift1_embedding(shifted1_x)
        shifted1_x = layers.Add()([shifted1_x, base_x])

        shifted2_x = shift2_embedding(shifted2_x)
        shifted2_x = layers.Add()([shifted2_x, shifted1_x])

        shifted3_x = shift3_embedding(shifted3_x)
        shifted3_x = layers.Add()([shifted3_x, shifted2_x])

        shifted4_x = shift4_embedding(shifted4_x)
        shifted4_x = layers.Add()([shifted4_x, shifted3_x])

        model = tf.keras.models.Model(inputs=input_x, outputs=shifted4_x)

    return model
