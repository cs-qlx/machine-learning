from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def data_processing():
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    mnist = input_data.read_data_sets('./', one_hot=True)
    np.where(mnist.train.images > 0, 1, mnist.train.images)
    np.where(mnist.test.images > 0, 1, mnist.test.images)

    tf.logging.set_verbosity(old_v)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


