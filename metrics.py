import tensorflow as tf


def triplet_accuracy(a_pred, p_pred, n_pred, alpha, batch_size=32):
    positive_distance = tf.cast(
        tf.math.sqrt(tf.math.reduce_sum(tf.math.square(a_pred - p_pred), axis=-1)),
        dtype=tf.float32,
    )
    negative_distance = tf.cast(
        tf.math.sqrt(tf.math.reduce_sum(tf.math.square(a_pred - n_pred), axis=-1)),
        dtype=tf.float32,
    )
    total_correct = 0
    total_correct += tf.reduce_sum(tf.cast(positive_distance < alpha, tf.float32))
    total_correct += tf.reduce_sum(tf.cast(negative_distance > alpha, tf.float32))
    return 64 - total_correct
