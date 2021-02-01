import tensorflow as tf
from tensorflow.keras.regularizers import l2

def weight_init(mean=0, stddev=0.01):
    return tf.random_normal_initializer(mean=mean, stddev=stddev)


def bia_init(mean=0.5, stddev=0.01):
    return tf.random_normal_initializer(mean=mean, stddev=stddev)


def reg(factor=2e-4):
    return l2(factor)
