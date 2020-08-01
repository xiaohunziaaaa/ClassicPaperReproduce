# network used in GAN, including classifier and discriminator
import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dense, dropout

class g_conv(object):

    def __init__(self):
        self.name = 'g_conv'
        self.size = 28
        self.channel = 1

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            output = dense(inputs=z, units=128, activation='relu')
            output = dense(inputs=output, units=28*28, activation='relu')
            output = tf.reshape(tensor=output, shape=[-1, self.size, self.size, self.channel])
            return output

    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class d_conv(object):

    def __init__(self):
        self.name = 'd_conv'
        self.size = 28
        self.channel = 1

    def __call__(self, z, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = tf.layers.flatten(z)
            output = dense(inputs=output, units=128, activation='relu')
            output = dense(inputs=output, units=1, activation=None)
            return output

    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class cg_conv(object):

    def __init__(self):
        self.name = 'cg_conv'
        self.size = 28
        self.channel = 1

    def __call__(self, z, y):
        with tf.variable_scope(self.name) as scope:
            joint = tf.concat(values=[z, y], axis=1)
            output = dense(inputs=joint, units=128, activation='relu')
            output = dense(inputs=output, units=28*28, activation='relu')
            output = tf.reshape(tensor=output, shape=[-1, self.size, self.size, self.channel])
            return output

    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class cd_conv(object):

    def __init__(self):
        self.name = 'cd_conv'
        self.size = 28
        self.channel = 1

    def __call__(self, x, y, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            output = tf.layers.flatten(x)
            joint = tf.concat(values=[output, y], axis=1)
            output = dense(inputs=joint, units=128, activation='relu')
            output = dense(inputs=output, units=1, activation=None)
            return output

    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]