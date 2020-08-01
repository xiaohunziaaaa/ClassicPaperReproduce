import tensorflow as tf
import numpy as np
from nets import g_conv, d_conv
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# training paremeters
epochs = 1000
batch_size = 64
training_samples = 60000
total_steps = int (training_samples/batch_size)


# load input
(x_train, y_train), (x_test, y_test)=mnist.load_data()
x_train = x_train[0:training_samples]
y_train = y_train[0:training_samples]
x_train = np.reshape(a=x_train, newshape=[-1, 28, 28, 1])


def sample_Z(samples=128, dimension=100):
    Z = np.random.uniform(-1., 1., size=[samples, dimension])
    return Z


# construct models
# input placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
Z = tf.placeholder(dtype=tf.float32, shape=[None, 100])
# nets
Genrator = g_conv()
Discriminator = d_conv()
G_vars = Genrator.vars()
D_vars = Discriminator.vars()
G_sample = Genrator(Z)
D_real = Discriminator(X)
D_fake = Discriminator(G_sample, reuse=True)
G_vars = Genrator.vars()
D_vars = Discriminator.vars()
print(G_vars)
print(D_vars)
# loss
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
# optimizer
G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=G_loss, var_list=G_vars)
D_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=D_loss, var_list=D_vars)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')


i = 0
global_steps = 0
# Training network
# two steps train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    for step in range(total_steps):
        x_batch = x_train[step * batch_size: (step + 1) * batch_size]
        z_batch = sample_Z(batch_size, 100)
        # train Discriminator
        D_train_, D_loss_ = sess.run([D_optimizer, D_loss], feed_dict={X:x_batch, Z:z_batch})
        # train Generator
        z_batch = sample_Z(batch_size, 100)
        G_train_, G_loss_= sess.run([G_optimizer, G_loss], feed_dict={Z:z_batch})
        if global_steps % 500 == 0:
            print('Global steps {}, D_loss is {}, G_loss is {}'.format(global_steps, D_loss_, G_loss_))
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, 100)})
            fig = plot(samples)
            plt.savefig('GANout/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
        global_steps += 1