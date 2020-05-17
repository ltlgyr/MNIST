#!/bin/python3

import scipy.misc
import os 

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
# 1 。loaddata
# 2.test train distance
# 3. k个最近的图片5 500 1 -》500 train 4
# 4 k个最近的图片-》parse  centent label
# 5 label -> 数字
# 6 检测数据的统计
# load data
mnist = input_data.read_data_sets('../train', one_hot = True)
trainNum = 55000
testNum  = 10000
trainSize= 100
testSize = 10
k = 5
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


trainDataInput  = tf.placeholder(shape = [None,784], dtype = tf.float32)
trainLabelInput = tf.placeholder(shape = [None, 10], dtype = tf.float32)

testDataInput =  tf.placeholder(shape = [None,784], dtype = tf.float32)
testLabelInput =  tf.placeholder(shape = [None, 10], dtype = tf.float32)

# 距离计算
#维度扩展 expand_dims (input, dim, name=None)
# input 输入， dim 指定维度
f1 = tf.expand_dims(testDataInput,1)  
# subtract 减法
f2 = tf.subtract(trainDataInput,f1) # 784, num(784)
# sum 
#a = [[[1,2], [3,4]],                                                                                                                              |   
#     [[5,6], [7,8]]];
# axis轴
# axis 0 [[1,2]],[[3,4]],[[5,6]],[[7,8]]
#      1  [1,2],[3,4], [5,6], [7,8]
#      2   1,2,  3,4,   5,6,   7,8
# keepdims True :保持现在的维度，不掉括号
# reduction_indices 弃用
# keep_dims 弃用
# recuce 塌缩，归约  
# reduce_sum (input_tensor, axis = None, keepdims = None, name = None, reduction_indices = None, keep_dims = None)

f3 = tf.reduce_sum(tf.abs(f2), reduction_indices=2) # 完成数据累加 784 abs
# negative 取反
f4 = tf.negative (f3) # 取反
# 
f5,f6 = tf.nn.top_k(f4,k=5) #选取f4中最大的4个字。
f7 = tf.gather(trainLabelInput, f6)

f8 = tf.reduce_sum(f7, reduction_indices=1)
# 最大值，并记录下标
f9 = tf.argmax(f8, dimension =1)

# 5 * 500
with tf.Session() as sess:
	# f2 <- testdata 5 张图片
	p1 = sess.run(f1, feed_dict = {testDataInput:testData[0:testSize]})
	print('p1.shape = ', p1.shape)
	p2 = sess.run(f2, feed_dict = {trainDataInput:trainData, testDataInput:testData[0:testSize]})
	print('p2.shape = ', p2.shape)
	print('p2.shape = ', testData.shape, trainData.shape)
	p3 = sess.run(f3, feed_dict = {trainDataInput:trainData, testDataInput:testData[0:testSize]})
	print('p3 = ', p3.shape)
	print('p3[0,0] = ', p3[0,0]) # 130 451
	p4 = sess.run (f4, feed_dict ={trainDataInput:trainData, testDataInput:testData[0:testSize]})
	print('p4 = ', p4.shape)
	print('p4[0,0] = ', p4[0,0]) # 130 451
	p5,p6 = sess.run ((f5,f6), feed_dict ={trainDataInput:trainData, testDataInput:testData[0:testSize]})
	print('p5 = ', p5.shape)
	print('p6 = ', p6.shape)
	print('p5 = ', p5)
	print('p6 = ', p6)
	p7 = sess.run (f7, feed_dict ={trainDataInput:trainData, testDataInput:testData[0:testSize], trainLabelInput:trainLabel})
	print('p7 = ', p7.shape)
	print('p7 = ', p7)
	p8 = sess.run (f8, feed_dict ={trainDataInput:trainData, testDataInput:testData[0:testSize], trainLabelInput:trainLabel})
	print('p8 = ', p8.shape)
	print('p8 = ', p8)
	p9 = sess.run (f9, feed_dict ={trainDataInput:trainData, testDataInput:testData[0:testSize], trainLabelInput:trainLabel})
	print('p9 = ', p9.shape)
	print('p9 = ', p9)
	
	p10 = np.argmax (testLabel[0:testSize], axis=1)
	print('p10= ', p10)
j = 0
for i in range(0, testSize):
	if p10[i]  == p9[i]:
		j = j +1
print("ac = ", j * 100/testSize) 
