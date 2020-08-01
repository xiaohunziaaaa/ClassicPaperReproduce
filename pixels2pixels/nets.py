import tensorflow as tf
import numpy as np

# G_UNet use images as input, do not contain random variable which is contained in cGAN of paper Conditional GAN
class G_UNet(object):
    def __init__(self):
        self.name = 'G_UNet'

    def __call__(self, input_ph_x):
        with tf.variable_scope(self.name) as scope:
            # encoder
            s = 256             # ouput size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # input is 256 * 256 * 3
            # halve feature size each conv2d with kernel_size = 4, strides = [2,2]
            encoder1 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=input_ph_x, filters=16, padding='same',
                                                                       strides=[2, 2], kernel_size=4))  # 128*128*16
            encoder2 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=encoder1, filters=32, padding='same',
                                                                       strides=[2, 2], kernel_size=4)) # 64*64*32
            encoder3 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=encoder2, filters=64, padding='same',
                                                                       strides=[2, 2], kernel_size=4))  # 32*32*64
            encoder4 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=encoder3, filters=128, padding='same',
                                                                       strides=[2, 2], kernel_size=4))  # 16*16*128
            encoder5 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=encoder4, filters=128, padding='same',
                                                                       strides=[2, 2], kernel_size=4))  # 8*8*128
            encoder6 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=encoder5, filters=128, padding='same',
                                                                       strides=[2, 2], kernel_size=4))  # 4*4*128
            encoder7 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=encoder6, filters=128, padding='same',
                                                                       strides=[2, 2], kernel_size=4))  # 2*2*128
            encoder8 = tf.layers.batch_normalization(tf.layers.conv2d(inputs=encoder7, filters=128, padding='same',
                                                                       strides=[2, 2], kernel_size=4))  # 1*1*128

            # decoder
            decoder1 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(encoder8), filters=128,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder1 = tf.concat(values=[encoder7, decoder1], axis=3)

            decoder2 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(decoder1), filters=128,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder2 = tf.concat(values=[encoder6, decoder2], axis=3)

            decoder3 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(decoder2), filters=128,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder3 = tf.concat(values=[encoder5, decoder3], axis=3)

            decoder4 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(decoder3), filters=128,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder4 = tf.concat(values=[encoder4, decoder4], axis=3)

            decoder5 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(decoder4), filters=64,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder5 = tf.concat(values=[encoder3, decoder5], axis=3)

            decoder6 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(decoder5), filters=32,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder6 = tf.concat(values=[encoder2, decoder6], axis=3)

            decoder7 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(decoder6), filters=16,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder7 = tf.concat(values=[encoder1, decoder7], axis=3)

            decoder8 = tf.layers.conv2d_transpose(inputs=tf.nn.relu(decoder7), filters=3,
                                                  strides=[2, 2], kernel_size=4, padding='same')
            decoder8 = tf.nn.sigmoid(decoder8)

            return decoder8

    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class D_Patch(object):
    def __init__(self):
        self.name = 'D_Patch'

    def __call__(self, input_ph_y, input_ph_x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            input_ph = tf.concat(values=[input_ph_y, input_ph_x], axis=3)       # 256*256*6
            c1 = tf.layers.conv2d(inputs=input_ph, filters=16, strides=[2,2], activation='sigmoid', kernel_size=4)    #128*128*16
            c2 = tf.layers.conv2d(inputs=c1, filters=16, strides=[2,2], activation='sigmoid', kernel_size=4)    #64*64*16
            c3 = tf.layers.conv2d(inputs=c2, filters=32, strides=[2, 2], activation='sigmoid',  kernel_size=4)  # 32*32*32
            c4 = tf.layers.conv2d(inputs=c3, filters=1, strides=[2, 2], activation=None, kernel_size=4)   # 16*16*1
            c5 = tf.layers.flatten(c4)
            c5 = tf.reduce_mean(input_tensor=c4, axis=1)
            return c5

    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]