#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:10:11 2018

@author: ld
"""
import tensorflow as tf
import math
import random
import numpy as np
# import os
from tensorflow.contrib import layers
from tflearn import global_avg_pool
from scipy.misc import imresize

# from datetime import datetime
slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
initializer1 = tf.contrib.layers.xavier_initializer()
initializer = tf.contrib.layers.xavier_initializer()


def lrelu(x, leak=0.2, name="lrelu"):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def trainTestSplit(L, train_size):
    trainIndex = []
    testIndex = []
    for i in range(np.shape(L)[1]):
        index = np.where(L[:, i] == 1)
        index_ = index[0].tolist()
        l_index = random.sample(index_, int(len(index_) * train_size))
        trainIndex = trainIndex + l_index
        testIndex = testIndex + list(set(index_).difference(set(l_index)))
    return np.array(trainIndex), np.array(testIndex)


def flatten(x):
    return tf.contrib.layers.flatten(x)


def max_(x):
    if x > 0:
        return x
    else:
        return 0


def dropout(x, rate, is_training):
    return tf.layers.dropout(inputs=x, rate=rate, training=is_training)


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def concat(x, axis=1):
    return tf.concat(x, axis=axis)


def conv_cond_concat(x, y):
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)

    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


def batch_norm(x, is_training):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training)


def get_image(x, batch_size, img_h, img_w):
    image_batch = []
    for j in range(batch_size):
        data = x[j, :, :, :]
        data = np.array(data)
        data = data.astype('float32') / 127.5 - 1
        image_batch.append(data)
    return (image_batch)


def restruct_image(x, batch_size):
    image_batch = []
    for k in range(batch_size):
        data = x[k, :, :, :]
        data = (data + 1) * 127.5
        # data = np.clip(data,0,255).astype(np.uint8)
        image_batch.append(data)
    return (image_batch)


def get_image_vgg16(x, batch_size):
    image_batch = []
    for j in range(batch_size):
        data = x[j, :, :, :]
        data = np.array(data)
        data = imresize(data, (224, 224))
        image_batch.append(data)
    return (image_batch)


########################################################################################VGG##########################
# def vgg_16(net_in,train):
#      mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='mean_rgb')
#      net_in= net_in- mean
#      net = slim.repeat( net_in, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#      net = slim.max_pool2d(net, [2, 2], scope='pool1')
#      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#      net = slim.max_pool2d(net, [2, 2], scope='pool2')
#      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#      net = slim.max_pool2d(net, [2, 2], scope='pool3')
#      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#      net = slim.max_pool2d(net, [2, 2], scope='pool4')
#      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#      net = slim.max_pool2d(net, [2, 2], scope='pool5')
#      net = slim.conv2d(net, 4096, [7, 7], padding='SAME', scope='fc6')
##      net = slim.dropout(net, 0.5, scope='dropout6')
##      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
##      net = slim.dropout(net, 0.5, scope='dropout7')
##      net = slim.conv2d(net, 1000, [1, 1],
##                          activation_fn=None,
##                          normalizer_fn=None,
##                         scope='fc8')
#      shp =net.get_shape()  
#      flattened_shape = shp[1].value * shp[2].value * shp[3].value  
#      net = tf.reshape(net, [-1, flattened_shape], name='flatten')
#      return net

def fc_layers(net, num_label, train):
    #
    net1 = layers.fully_connected(net, 1000, activation_fn=tf.nn.relu, weights_initializer=initializer1)
    #    net1 = dropout(net1, rate=0.5,is_training=train)
    net2 = layers.fully_connected(net1, 200, activation_fn=None, weights_initializer=initializer1)
    net3 = tf.nn.tanh(net2)
    #    net3 = dropout(net3, rate=0.5,is_training=train)
    net4 = layers.fully_connected(net3, num_label, activation_fn=None, weights_initializer=initializer1)
    net5 = tf.nn.softmax(net4)

    return net2, net4, net5


#####################################################################################################
def Q(x, num_label, train):  # Non modal image
    h1 = layers.fully_connected(x, num_outputs=1000, activation_fn=tf.nn.relu, weights_initializer=initializer1)
    # h1=dropout(h1, rate=0.5,is_training=train)
    h2 = layers.fully_connected(h1, num_outputs=200, activation_fn=None, weights_initializer=initializer1)
    h3 = tf.nn.tanh(h2)
    # h3=dropout(h3, rate=0.5,is_training=train)
    h4 = layers.fully_connected(h3, num_outputs=num_label, activation_fn=None, weights_initializer=initializer1)
    h5 = tf.nn.softmax(h4)
    return h2, h4, h5


def get_size(size, stride, name='get_size'):
    return int(math.ceil(float(size) / float(stride)))


def generator(x, y1, y2, train):
    y = tf.concat([y1, y2], axis=1)
    h = tf.concat([x, y], axis=1)
    h = layers.fully_connected(h, 1024, activation_fn=tf.nn.relu, weights_initializer=initializer)

    h = layers.dropout(h, keep_prob=0.8, is_training=train)
    h = layers.fully_connected(h, 64 * 7 * 7 * 2, activation_fn=tf.nn.relu, weights_initializer=initializer)
    h = tf.reshape(h, [-1, 7, 7, 64 * 2])

    h = layers.dropout(h, keep_prob=0.8, is_training=train)
    h = layers.conv2d_transpose(h, 64 * 1, 5, stride=2, padding='SAME', activation_fn=tf.nn.relu,
                                weights_initializer=initializer)

    h = layers.dropout(h, keep_prob=0.8, is_training=train)
    h = layers.conv2d_transpose(h, 1, 5, stride=2, padding='SAME', activation_fn=tf.nn.sigmoid,
                                weights_initializer=initializer)
    return h


def discriminator_xx(x1, x2, train):
    net1 = tf.reshape(x1, [-1, 28, 28, 1])
    net2 = tf.reshape(x2, [-1, 28, 28, 1])
    net = tf.concat([net1, net2], axis=1)
    # pdb.set_trace()
    net = layers.conv2d(net, 32, 5, stride=2, activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 64, 5, stride=2, activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer())
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID', activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer())
    # net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    return layers.fully_connected(net, 1, activation_fn=None,
                                  weights_initializer=tf.contrib.layers.xavier_initializer())


def discriminator(x, y1, y2, train):
    y = tf.concat([y1, y2], axis=1)
    h = layers.conv2d(x, 32, 5, stride=2, activation_fn=tf.nn.relu, weights_initializer=initializer)
    h = layers.conv2d(h, 64, 3, stride=2, activation_fn=tf.nn.relu, weights_initializer=initializer)

    h = layers.flatten(h)
    h = tf.concat([h, y], axis=1)
    h = layers.dropout(h, keep_prob=0.9, is_training=train)
    h = layers.fully_connected(h, 128, activation_fn=lrelu, weights_initializer=initializer)
    h = layers.dropout(h, keep_prob=0.9, is_training=train)
    return layers.fully_connected(h, 1, activation_fn=None, weights_initializer=initializer)
