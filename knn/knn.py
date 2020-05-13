#!/bin/python3

import scipy.misc
import os 

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# load data
mnist = input_data.read_data_sets('../train', one_hot = True)

trainNum = 55000
testNum  = 10000
trainSize= 500
testSize = 5
k = 4
trainIndex = np.random.choice(trainNum, trainSize,replace = False)
testIndex  = np.random.choice(testNum, testSize, replace= False)

trainData  = mnist.train.images[trainIndex]
trainLabel = mnist.train.labels[trainIndex]
testData   = mnist.test.images [testIndex]
testLabel  = mnist.test.labels [testIndex]
print('trainData.shape = ', trainData.shape)   # 500*784
print('trainLabel.shape = ', trainLabel.shape) # 500*10
print('testData.shape = ', testData.shape)     # 5*784
print('testLabel.shape = ', testLabel.shape)   # 5*10
print('testLabel', testLabel)                  # 
trainDataInput = tf.placeholder(shape = [None,784], dtype = tf.float32)
trainLabelInput = tf.placeholder(shape=[None, 10], dtype = tf.float32)

testDataInput = tf.placeholder(shape = [None,784], dtype = tf.float32)
testLabelInput = tf.placeholder(shape=[None, 10], dtype = tf.float32)

# 距离计算

