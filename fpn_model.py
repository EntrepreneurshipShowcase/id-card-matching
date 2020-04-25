import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from losses import triplet_loss
from metrics import triplet_accuracy
from tensorflow.keras.applications import resnet

import fpn
import resnet50

EMBEDDING_LAYER_DIM = 256


class Embedding(layers.Layer):
    def __init__(self):
        super(Embedding, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.conv1 = layers.Conv2D(EMBEDDING_LAYER_DIM, (1, 1))
        self.bn = layers.BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
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
    def __init__(self):
        super(Siamese, self).__init__()
        self.global_pool = layers.GlobalMaxPool2D()
        self.embedding_p3 = Embedding()
        self.embedding_p4 = Embedding()
        self.embedding_p5 = Embedding()
        self.embedding_p2 = Embedding()
        self.embedding_p6 = Embedding()
        # self.backbone = resnet50.ResNet(101)
        # self.backbone.load_weights("./faster_rcnn.h5", by_name=True)
        base_model = resnet.ResNet50(include_top=False, weights="imagenet")
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=(base_model.get_layer("conv2_block3_out").output, base_model.get_layer("conv3_block4_out").output, base_model.get_layer("conv4_block6_out").output, base_model.output))
        # self.backbone.trainable = False
        self.neck = fpn.FPN(
            name='fpn')

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
    siamese = Siamese()
    inp_shape = (224, 224, 3)
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

        p2_a, p3_a, p4_a, p5_a, p6_a = siamese(input_a, training=training)
        p2_p, p3_p, p4_p, p5_p, p6_p = siamese(input_p, training=training)
        p2_n, p3_n, p4_n, p5_n, p6_n = siamese(input_n, training=training)
        
        p2_a, p2_p, p2_n = TripletLoss(margin=4)(p2_a, p2_p, p2_n)
        p3_a, p3_p, p3_n = TripletLoss(margin=7)(p3_a, p3_p, p3_n)
        p4_a, p4_p, p4_n = TripletLoss(margin=10)(p4_a, p4_p, p4_n)
        p5_a, p5_p, p5_n = TripletLoss(margin=13)(p5_a, p5_p, p5_n)
        p6_a, p6_p, p6_n = TripletLoss(margin=16)(p6_a, p6_p, p6_n)
        p6 = layers.Concatenate()([p6_a, p6_p, p6_n])
        model = tf.keras.models.Model(
            inputs=[input_a, input_p, input_n], outputs=p6
        )
    else:
        input_x = layers.Input(inp_shape, name="input")
        _, _, _, _, p6_x = siamese(input_x, training=training)
        model = tf.keras.models.Model(inputs=input_x, outputs=p6_x)

    return model