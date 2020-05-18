#!/bin/python3
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets('../train', one_hot = True)

imageInput  = tf.placeholder(shape = [None,784], dtype = tf.float32)
labelInput = tf.placeholder(shape = [None, 10], dtype = tf.float32)
imageInputReshape = tf.reshapeh(imageInput,[-1,28,28,1]

w0 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev =0.1))
b0 = tf.Variable(tf.constant(0.1,shape=[32])

layer1 = tf.nn.relu(tf.nn.conv2d(imageInputReshape, w0, strides= [1,1,1,1],padding = 'SAME') +b0)


