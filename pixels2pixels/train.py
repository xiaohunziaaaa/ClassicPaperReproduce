import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import utils
import nets
# parameters
epochs = 100
batch_size = 1


# load input
input_path = '../Dataset/cityscapes'
raw_train_images, raw_test_images = utils.load_images(input_path)
# cut raw image into two parts, left part is org images, right part is annotation
# in this experiment, it is a label-to-street-scene task , so condition is annotation, output is org images
b_train, a_train = utils.imdivandpre(raw_train_images)
b_test, a_test = utils.imdivandpre(raw_test_images)
a_test_single = a_test[0]
a_test_single = np.reshape(a=a_test_single, newshape=[-1, 256, 256, 3])
total_steps = int(len(a_train)/batch_size)

# construct model
A = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])         # A: annotated images(condition)
B = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])         # B: real images

Generator = nets.G_UNet()
G_sample = Generator(A)                                                   # g: Generator
G_vars = Generator.vars()

Discriminator = nets.D_Patch()
D_real = Discriminator(B, A)                                              # d_real: real images
D_fake = Discriminator(G_sample, A, reuse=True)                           # d_fake: fake images(generated images )
D_vars = Discriminator.vars()

G_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
G_loss_L1 = tf.reduce_mean(tf.abs(B - G_sample))
G_loss = G_loss_GAN + G_loss_L1
D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=G_loss, var_list=G_vars)
D_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=D_loss, var_list=D_vars)

# train
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
global_step = 0
for epoch in range(epochs):
    for step in range(total_steps):
        a_batch = a_train[step * batch_size: (step + 1) * batch_size]/255.0
        b_batch = b_train[step * batch_size: (step + 1) * batch_size]/255.0

        # update Discriminator
        _, D_loss_ = sess.run([D_optimizer, D_loss], feed_dict={A: a_batch, B: b_batch})
        print('Epoch {}, step {}, D_loss is {}, '.format(epoch, step, D_loss_), end='')
        # update Generator
        _, G_loss_ = sess.run([G_optimizer, G_loss], feed_dict={A: a_batch, B: b_batch })
        print('G_loss is {}'.format(G_loss_))

        if global_step % 100 == 0:
            # do testing
            a_input = a_test_single/255.0
            b_output = sess.run(G_sample, feed_dict={A: a_input})
            out = np.concatenate((a_input, b_output), axis=2)
            out = np.reshape(a=out, newshape=[256, 512, 3])
            plt.imsave(fname='./out/{}.jpg'.format(global_step), arr=out)
        global_step += 1



