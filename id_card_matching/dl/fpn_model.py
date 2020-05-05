import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

from dl.losses import triplet_loss
from dl.metrics import triplet_accuracy
from tensorflow.keras.applications import resnet

import dl.fpn as fpn

EMBEDDING_LAYER_DIM = 128


class Embedding(layers.Layer):
    def __init__(self):
        super(Embedding, self).__init__()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.conv1 = layers.Conv2D(EMBEDDING_LAYER_DIM*2, (2, 2))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = layers.Activation("relu")(x)
        x = self.conv2(x)
        x = self.bn2(x)
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

class Siamese(layers.Layer):
    def __init__(self, input_shape):
        super(Siamese, self).__init__()
        self.embedding_p3 = Embedding()
        self.embedding_p4 = Embedding()
        self.embedding_p5 = Embedding()
        self.embedding_p2 = Embedding()
        self.embedding_p6 = Embedding()
        self.inp_shape = input_shape
        # self.backbone = resnet50.ResNet(101)
        # self.backbone.load_weights("./faster_rcnn.h5", by_name=True)
        base_model = resnet.ResNet50(include_top=False, input_shape=input_shape, weights="imagenet")
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=(base_model.get_layer("conv2_block3_out").output, base_model.get_layer("conv3_block4_out").output, base_model.get_layer("conv4_block6_out").output, base_model.output))
        # self.backbone.trainable = False
        self.neck = fpn.FPN(
            name='fpn')
    def get_config(self):
        return {"base_input": self.inp_shape}

    def call(self, inputs, training):
        C2_a, C3_a, C4_a, C5_a = self.backbone(inputs, training=training)
        P2_a, P3_a, P4_a, P5_a, P6_a = self.neck([C2_a, C3_a, C4_a, C5_a], training=training)

        p2_a = self.embedding_p2(P2_a)

        p3_a = self.embedding_p3(P3_a)
        p3_a = layers.Add()([p2_a, p3_a])

        p4_a = self.embedding_p4(P4_a)
        p4_a = layers.Add()([p3_a, p4_a])

        p5_a = self.embedding_p5(P5_a)
        p5_a = layers.Add()([p4_a, p5_a])

        p6_a = self.embedding_p6(P6_a)
        p6_a = layers.Add()([p5_a, p6_a])

        return p2_a, p3_a, p4_a, p5_a, p6_a


def get_siamese_model(training=True):
    inp_shape = (96, 96, 3)
    base_model = resnet.ResNet50(include_top=False, input_shape=inp_shape, weights="imagenet")
    backbone = tf.keras.Model(inputs=base_model.input, outputs=(base_model.get_layer("conv2_block3_out").output, base_model.get_layer("conv3_block4_out").output, base_model.get_layer("conv4_block6_out").output, base_model.output))

    embedding_p3 = tf.keras.Sequential([layers.Conv2D(EMBEDDING_LAYER_DIM*2, layers.BatchNormalization(), (2, 2)), layers.Activation("relu"), layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1)), layers.BatchNormalization(), layers.Activation("relu"), layers.GlobalAveragePooling2D()])
    embedding_p4 = tf.keras.Sequential([layers.Conv2D(EMBEDDING_LAYER_DIM*2, layers.BatchNormalization(), (2, 2)), layers.Activation("relu"), layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1)), layers.BatchNormalization(), layers.Activation("relu"), layers.GlobalAveragePooling2D()])
    embedding_p5 = tf.keras.Sequential([layers.Conv2D(EMBEDDING_LAYER_DIM*2, layers.BatchNormalization(), (2, 2)), layers.Activation("relu"), layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1)), layers.BatchNormalization(), layers.Activation("relu"), layers.GlobalAveragePooling2D()])
    embedding_p2 = tf.keras.Sequential([layers.Conv2D(EMBEDDING_LAYER_DIM*2, layers.BatchNormalization(), (2, 2)), layers.Activation("relu"), layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1)), layers.BatchNormalization(), layers.Activation("relu"), layers.GlobalAveragePooling2D()])
    embedding_p6 = tf.keras.Sequential([layers.Conv2D(EMBEDDING_LAYER_DIM*2, layers.BatchNormalization(), (2, 2)), layers.Activation("relu"), layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1)), layers.BatchNormalization(), layers.Activation("relu"), layers.GlobalAveragePooling2D()])
    # backbone = resnet50.ResNet(101)
    # backbone.load_weights("./faster_rcnn.h5", by_name=True)
    # backbone.trainable = False
    neck = fpn.FPN(
        name='fpn')
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

        C2_a, C3_a, C4_a, C5_a = backbone(input_a, training=training)
        P2_a, P3_a, P4_a, P5_a, P6_a = neck([C2_a, C3_a, C4_a, C5_a], training=training)

        p2_a = embedding_p2(P2_a)

        p3_a = embedding_p3(P3_a)
        p3_a = layers.Add()([p2_a, p3_a])

        p4_a = embedding_p4(P4_a)
        p4_a = layers.Add()([p3_a, p4_a])

        p5_a = embedding_p5(P5_a)
        p5_a = layers.Add()([p4_a, p5_a])

        p6_a = embedding_p6(P6_a)
        p6_a = layers.Add()([p5_a, p6_a])


        C2_p, C3_p, C4_p, C5_p = backbone(input_p, training=training)
        P2_p, P3_p, P4_p, P5_p, P6_p = neck([C2_p, C3_p, C4_p, C5_p], training=training)

        p2_p = embedding_p2(P2_p)

        p3_p = embedding_p3(P3_p)
        p3_p = layers.Add()([p2_p, p3_p])

        p4_p = embedding_p4(P4_p)
        p4_p = layers.Add()([p3_p, p4_p])

        p5_p = embedding_p5(P5_p)
        p5_p = layers.Add()([p4_p, p5_p])

        p6_p = embedding_p6(P6_p)
        p6_p = layers.Add()([p5_p, p6_p])

        C2_n, C3_n, C4_n, C5_n = backbone(input_n, training=training)
        P2_n, P3_n, P4_n, P5_n, P6_n = neck([C2_n, C3_n, C4_n, C5_n], training=training)

        p2_n = embedding_p2(P2_n)

        p3_n = embedding_p3(P3_n)
        p3_n = layers.Add()([p2_n, p3_n])

        p4_n = embedding_p4(P4_n)
        p4_n = layers.Add()([p3_n, p4_n])

        p5_n = embedding_p5(P5_n)
        p5_n = layers.Add()([p4_n, p5_n])

        p6_n = embedding_p6(P6_n)
        p6_n = layers.Add()([p5_n, p6_n])
        
        p2_a, p2_p, p2_n = TripletLoss(margin=4)(p2_a, p2_p, p2_n)
        p3_a, p3_p, p3_n = TripletLoss(margin=7)(p3_a, p3_p, p3_n)
        p4_a, p4_p, p4_n = TripletLoss(margin=10)(p4_a, p4_p, p4_n)
        p5_a, p5_p, p5_n = TripletLoss(margin=13)(p5_a, p5_p, p5_n)
        p6_a, p6_p, p6_n = TripletLoss(margin=16)(p6_a, p6_p, p6_n)
        # p2_a, p2_p, p2_n = TripletAccuracy(margin=4)(p2_a, p2_p, p2_n)
        # p3_a, p3_p, p3_n = TripletAccuracy(margin=7)(p3_a, p3_p, p3_n)
        # p4_a, p4_p, p4_n = TripletAccuracy(margin=10)(p4_a, p4_p, p4_n)
        # p5_a, p5_p, p5_n = TripletAccuracy(margin=13)(p5_a, p5_p, p5_n)
        p6_a, p6_p, p6_n = TripletAccuracy(margin=85)(p6_a, p6_p, p6_n)
        p6 = layers.Concatenate()([p6_a, p6_p, p6_n])
        model = tf.keras.models.Model(
            inputs=[input_a, input_p, input_n], outputs=p6
        )
    else:
        C2_x, C3_x, C4_x, C5_x = backbone(input_x, training=training)
        P2_x, P3_x, P4_x, P5_x, P6_x = neck([C2_x, C3_x, C4_x, C5_x], training=training)

        p2_x = embedding_p2(P2_x)

        p3_x = embedding_p3(P3_x)
        p3_x = layers.Add()([p2_x, p3_x])

        p4_x = embedding_p4(P4_x)
        p4_x = layers.Add()([p3_x, p4_x])

        p5_x = embedding_p5(P5_x)
        p5_x = layers.Add()([p4_x, p5_x])

        p6_x = embedding_p6(P6_x)
        p6_x = layers.Add()([p5_x, p6_x])
        model = tf.keras.models.Model(inputs=input_x, outputs=p6_x)

    return model
