#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:47:36 2018

@author: ldd
"""
from sklearn import svm
import tensorflow as tf

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
# from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile
import matplotlib.pyplot as plt
import scipy.misc
import matplotlib
from sklearn import preprocessing

matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
from model_minist import Q, generator, discriminator, get_image, restruct_image, fc_layers, get_image_vgg16, vgg_16, \
    discriminator_xx
import h5py
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# import math
from scipy.io import loadmat
import scipy.io as sio

ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
handwriten_char = loadmat("/data1/home/ldd/ld_project/PR/data/braindata/handwriten_char_28x28_S3.mat")
S_train = handwriten_char['fmriTrn']
S_test = handwriten_char['fmriTest']
X_train = handwriten_char['stimTrn']
X_test = handwriten_char['stimTest']
label_train_paired = handwriten_char['labelTrn']
label_test_paired = handwriten_char['labelTest']
unpaired = loadmat('/data1/home/ldd/ld_project/PR/data/braindata/unpaired_brains.mat')
X_unpaired = unpaired['unpaired_brains_data_28x28']
label_unpaired = unpaired['unpaired_brains_label']
# Voxel selection
voxel_idx = loadmat("/data1/home/ldd/lddecode/selected_voxel_idx_S3_char.mat")
select_voxel_idx = voxel_idx['selected_idx']
select_voxel_idx = np.reshape(select_voxel_idx, select_voxel_idx.shape[1])
S_train = S_train[:, select_voxel_idx]
S_test = S_test[:, select_voxel_idx]
label_train = label_train_paired
label_test = label_test_paired
min_max_scaler_1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(S_train)
S_train = min_max_scaler_1.transform(S_train)
S_test = min_max_scaler_1.transform(S_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_unpaired = X_unpaired.astype('float32')
X_train[X_train > 1.0] = 1.0
X_train[X_train <= 0.0] = 0.0
X_test[X_test > 1.0] = 1.0
X_test[X_test <= 0.0] = 0.0
X_unpaired[X_unpaired > 1.0] = 1.0
X_unpaired[X_unpaired <= 0.0] = 0.0

X_paired_28 = X_train
X_paired_28 = np.reshape(X_paired_28, [-1, 28, 28, 1])
X_test_28 = np.reshape(X_test, [-1, 28, 28, 1])
X_unpaired_28 = np.reshape(X_unpaired, [-1, 28, 28, 1])

from keras.utils import np_utils

nb_classes = len(np.unique(label_train_paired))
label_train_paired = np_utils.to_categorical(label_train_paired - 1, nb_classes)
nb_classes1 = len(np.unique(label_test_paired))
label_test_paired = np_utils.to_categorical(label_test_paired - 1, nb_classes1)
nb_classes_1 = len(np.unique(label_unpaired))
label_unpaired = np_utils.to_categorical(label_unpaired - 1, nb_classes_1)

f_dim = S_train.shape[1]
num_train = label_unpaired.shape[0]
# n_epochs = 1000


mb_size = 64
num_label = 6
num_train = label_unpaired.shape[0]
z_dim = 50
smooth = 0.1


def sample_SITL(S, I, T, L, size):
    start_idx = np.random.randint(0, S.shape[0] - size)
    return S[start_idx:start_idx + size], I[start_idx:start_idx + size], T[start_idx:start_idx + size], L[
                                                                                                        start_idx:start_idx + size]


def sample_XYZ(X, Y, Z, size):
    start_idx = np.random.randint(0, X.shape[0] - size)
    return X[start_idx:start_idx + size], Y[start_idx:start_idx + size], Z[start_idx:start_idx + size]


#########################################################################################
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


###########################################################################################
def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.rot90(np.fliplr(sample.reshape(28, 28))), cmap='hot')
    return fig


def log(x):
    return tf.log(x + 1e-8)


###############################################################################
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(loc=tf.zeros(shape), scale_diag=tf.ones(shape), **kwargs)), tf.float32)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def encoding_net_y2h(x, reuse, training):
    with tf.variable_scope("Q", reuse=reuse):
        h2, h4, h5 = Q(x, num_label, training)
    return h2, h4, h5


# def encoding_net_vgg16(x, reuse, training):
#    with tf.variable_scope("vgg_16", reuse=reuse):
#        net= vgg_16(x,training)
#    return net
def encoding_densenet(x, reuse, training):
    with tf.variable_scope("fc_layers", reuse=reuse):
        net2, net4, net5 = fc_layers(x, num_label, training)
    return net2, net4, net5


def generator_net(fea, y1, y2, reuse, training):
    with tf.variable_scope("generator", reuse=reuse):
        x = generator(fea, y1, y2, training)
    return x


def discriminator_net_xx(x, x1, reuse, training):
    with tf.variable_scope("discriminator_xx", reuse=reuse):
        log_d = discriminator_xx(x, x1, training)
    return tf.squeeze(log_d, squeeze_dims=[1])


def discriminator_net(x, y1, y2, reuse, training):
    with tf.variable_scope("discriminator", reuse=reuse):
        log_d = discriminator(x, y1, y2, training)
    return tf.squeeze(log_d, squeeze_dims=[1])


############################################
tf.reset_default_graph()
training = tf.placeholder(tf.bool, [])
I = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
I_unpaired = tf.placeholder(tf.float32, [None, 28 * 28 * 1])
X_unpaired_64 = tf.placeholder(tf.float32, [None, 28, 28, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
z1 = tf.placeholder(tf.float32, shape=[None, z_dim])
Y = tf.placeholder(tf.float32, shape=[None, f_dim])
l = tf.placeholder(tf.float32, shape=[None, num_label])
l_u = tf.placeholder(tf.float32, shape=[None, num_label])
# ll=tf.placeholder(tf.float32, shape=[None, mb_size])
# fea=tf.placeholder(tf.float32, shape = [None,X_dim], name='image_feature')
image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='image')  # may be real_paired image?
sim = tf.placeholder(tf.float32, shape=(None, 10))  # Label
sim_u = tf.placeholder(tf.float32, shape=(None, mb_size))  # Label
#############################################################################
h2, h4, h5 = encoding_net_y2h(Y, False, training)  # h2 = fc2 (EEG latent), h4 = before softmax, h5 = after softmax
net2, net4, net5 = encoding_densenet(I, False, training)  # net2 = fc7 (Image latent)
net2_unpaired, net4_unpaired, net5_unpaired = encoding_densenet(I_unpaired, True, training)

##################################

h_paired = tf.nn.tanh(h2)  # h_paired is eeg latent feed to generator
G = generator_net(z, h_paired, h5, False, training)
# net_G=encoding_net_vgg16(G_224,True,training)
# net2_G,net4_G,net5_G=encoding_densenet(net_G,True, training)
d_real = discriminator_net(image_input, h_paired, h5, False, training)  # paired stuff
d_fake = discriminator_net(G, h_paired, h5, True, training)             # ^same
xx_logit_real = discriminator_net_xx(image_input, image_input, False, training)
xx_logit_fake = discriminator_net_xx(image_input, G, True, training)

h_unpaired = tf.nn.tanh(net2_unpaired)
G_unpaired = generator_net(z1, h_unpaired, net5_unpaired, True, training)
# net_G_unpaired=encoding_net_vgg16(G_unpaired_224,True,training)
# net2_G_unpaired,net4_G_unpaired,net5_G_unpaired=encoding_densenet(net_G_unpaired,True, training)
d_real_unpaired = discriminator_net(X_unpaired_64, h_unpaired, net5_unpaired, True, training)
d_fake_unpaired = discriminator_net(G_unpaired, h_unpaired, net5_unpaired, True, training)
xx_logit_real_u = discriminator_net_xx(X_unpaired_64, X_unpaired_64, True, training)
xx_logit_fake_u = discriminator_net_xx(X_unpaired_64, G_unpaired, True, training)
###################################################################################
theta_y = 0.5 * tf.matmul(tf.nn.tanh(h2), tf.transpose(tf.nn.tanh(net2)))  # Theta_y is sigma (single number)
logloss_y = -tf.reduce_sum(tf.multiply(sim, theta_y) - log(1.0 + tf.exp(theta_y)))  #
loss_y = tf.div(logloss_y, float(10 * 10))

theta_x = 0.5 * tf.matmul(tf.nn.tanh(h2), tf.transpose(tf.nn.tanh(net2_unpaired)))
logloss_x = -tf.reduce_sum(tf.multiply(sim_u, theta_x) - log(1.0 + tf.exp(theta_x)))
loss_x = tf.div(logloss_x, float(10 * mb_size))
recon_loss = loss_x + loss_y
classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=l)) + tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=net4, labels=l)) \
                      + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net4_unpaired, labels=l_u))
x_sigmoid_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=xx_logit_real, labels=tf.ones_like(xx_logit_real)) * (
        1 - smooth)
x_sigmoid_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=xx_logit_fake, labels=tf.zeros_like(xx_logit_fake))
x_sigmoid_real_I = tf.nn.sigmoid_cross_entropy_with_logits(logits=xx_logit_real_u,
                                                           labels=tf.ones_like(xx_logit_real_u)) * (1 - smooth)
x_sigmoid_fake_II = tf.nn.sigmoid_cross_entropy_with_logits(logits=xx_logit_fake_u,
                                                            labels=tf.zeros_like(xx_logit_fake_u))

d_paired_loss = 1 * tf.reduce_mean(x_sigmoid_real + x_sigmoid_fake) + tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)) * (
            1 - smooth)) + tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
d_unpaired_loss = 1 * tf.reduce_mean(x_sigmoid_real_I + x_sigmoid_fake_II) + tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_unpaired, labels=tf.ones_like(d_real_unpaired)) * (
            1 - smooth)) \
                  + tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_unpaired, labels=tf.zeros_like(d_fake_unpaired)))

Discriminator_loss_paired = 1 * d_paired_loss
Discriminator_loss_unpaired = d_unpaired_loss
Discriminator_loss = 1 * Discriminator_loss_paired + Discriminator_loss_unpaired

xx_sigmoid_fake1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=xx_logit_fake, labels=tf.ones_like(xx_logit_fake)) * (
        1 - smooth)
xx_sigmoid_fake2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=xx_logit_fake_u,
                                                           labels=tf.ones_like(xx_logit_fake_u)) * (1 - smooth)

g_paired_loss = 1 * tf.reduce_mean(xx_sigmoid_fake1) + tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)) * (1 - smooth))
g_unpaired_loss = 1 * tf.reduce_mean(xx_sigmoid_fake2) + tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_unpaired, labels=tf.ones_like(d_fake_unpaired)) * (
            1 - smooth))
Generator_loss_paired = 1 * g_paired_loss
Generator_loss_unpaired = 1 * g_unpaired_loss
Generator_loss = 1 * Generator_loss_paired + Generator_loss_unpaired

classification_loss_1 = 1 * recon_loss + 1 * classification_loss

###########################################################################
train_vars = tf.trainable_variables()
# vgg_16_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vgg_16")

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
Q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Q")
fc_layers_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc_layers")
d_xx_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_xx")
#############################################################
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#    G_solver =  tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.4).minimize(Generator_loss1, var_list = g_vars+ Q_vars)
#    D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.4).minimize(Discriminator_loss, var_list = d_vars)
lr_d = tf.placeholder(tf.float32, shape=[])
D_solver = tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(Discriminator_loss, var_list=d_vars + d_xx_vars)
G_solver = tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(Generator_loss, var_list=g_vars)
lr_c = tf.placeholder(tf.float32, shape=[])
classfication_solver = tf.train.AdamOptimizer(lr_c, beta1=0.5).minimize(classification_loss_1,
                                                                        var_list=Q_vars + fc_layers_vars)
saver = tf.train.Saver()

##############################################################################
if not os.path.exists('minist/'):
    os.makedirs('minist/')
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)
i = 0
disc_steps = 1
gen_steps = 2
learning_rate = 1e-4
lr_D = 0.0001
lr = 0.0001
n_epochs = 80
for idx in range(n_epochs):
    #    lr_D = learning_rate/(idx+1.0)
    if idx > 0:
        lr = lr * 0.4
    if idx > 60:
        lr_D = lr_D * 0.95
    for it in range(0, num_train // mb_size):
        #
        class_step = 1
        s_t_mb, i_t_mb, ii_t_mb, l_t_mb = sample_SITL(S_train, X_paired_28, X_train, label_train_paired, 10)
        i_64_mb, i_224_mb, l_tu_mb = sample_XYZ(X_unpaired_28, X_unpaired, label_unpaired, mb_size)
        z_mb = sample_Z(10, z_dim)
        z_mb_1 = sample_Z(mb_size, z_dim)
        ll_mb = np.dot(l_t_mb, l_t_mb.T)
        ll_mbu = np.dot(l_t_mb, l_tu_mb.T)
        for m in range(class_step):
            _, recon_loss_curr, classification_loss_curr = sess.run(
                [classfication_solver, recon_loss, classification_loss],
                feed_dict={training: True, I: ii_t_mb, Y: s_t_mb, image_input: i_t_mb, l: l_t_mb, I_unpaired: i_224_mb,
                           X_unpaired_64: i_64_mb, l_u: l_tu_mb, z: z_mb, z1: z_mb_1, lr_c: lr, lr_d: lr_D, sim: ll_mb,
                           sim_u: ll_mbu})
        for k in range(disc_steps):
            _, D_loss_curr = sess.run([D_solver, Discriminator_loss],
                                      feed_dict={training: True, I: ii_t_mb, Y: s_t_mb, image_input: i_t_mb, l: l_t_mb,
                                                 I_unpaired: i_224_mb, X_unpaired_64: i_64_mb, l_u: l_tu_mb, z: z_mb,
                                                 z1: z_mb_1, lr_c: lr, lr_d: lr_D, sim: ll_mb, sim_u: ll_mbu})
        for j in range(gen_steps):
            _, G_loss_curr = sess.run([G_solver, Generator_loss],
                                      feed_dict={training: True, I: ii_t_mb, Y: s_t_mb, image_input: i_t_mb, l: l_t_mb,
                                                 I_unpaired: i_224_mb, X_unpaired_64: i_64_mb, l_u: l_tu_mb, z: z_mb,
                                                 z1: z_mb_1, lr_c: lr, lr_d: lr_D, sim: ll_mb, sim_u: ll_mbu})

        if it % 100 == 0:
            print(
                'epoch: {};Iter: {};D_loss: {:.4}; G_loss: {:.4};recon_loss: {:.4};classification_loss1: {:.4}'.format(
                    idx, it, D_loss_curr, G_loss_curr, recon_loss_curr, classification_loss_curr))

            #######################################################################################

            z_test = sample_Z(30, z_dim)
            #         z_test1=sample_Z(16,z_dim)
            samples, labels = sess.run([G, h5], feed_dict={Y: S_test, z: z_test, training: False})
            #         samples1,labels1= sess.run([G,h5],feed_dict = {Y:s_train_mb,z:z_test1,training: False})
            #         samples_1=(samples+1.)/2.
            #         samples2=(i_test_mb+1.)/2
            #         samples2=(samples1+1.)/2.
            samples_a = np.vstack([X_test_28, samples])
            fig = plot(samples_a)
            plt.savefig('minist/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
        #         saver.save(sess,'ckpt/mnist.ckpt',global_step=i)
