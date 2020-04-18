import tensorflow as tf


def triplet_loss(a_pred, p_pred, n_pred, margin):
    # embedding_size = y_pred.shape[-1] // 3
    # ind = int(embedding_size * 2)
    # a_pred = y_pred[:, :embedding_size]
    # p_pred = y_pred[:, embedding_size:ind]
    # n_pred = y_pred[:, ind:]
    a_pred = tf.cast(a_pred, tf.float32)
    p_pred = tf.cast(p_pred, tf.float32)
    n_pred = tf.cast(n_pred, tf.float32)

    positive_distance = tf.math.sqrt(
        tf.math.reduce_sum(tf.math.square(a_pred - p_pred), axis=-1)
    )

    negative_distance = tf.math.sqrt(
        tf.math.reduce_sum(tf.math.square(a_pred - n_pred), axis=-1)
    )

    loss = tf.math.maximum(0.0, positive_distance - negative_distance + margin)
    return loss
