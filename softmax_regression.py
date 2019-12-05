#j -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./train', one_hot=True)


'''
    #构建运算图
'''
# X Y 都是占位符 占位而已 不表示具体的数据 
x = tf.placeholder("float",[None,784]) # 图像的大小为784;None表示第一个维度可以是任意长度

# 一个Variable代表一个可修改的张量,它们可以用于计算输入值，也可以在计算中被修改
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

# 计算交叉熵
y_ = tf.placeholder("float", [None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))

# 梯度下降算法（gradient descent algorithm）
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
init = tf.global_variables_initializer()

# 在一个Session里面启动我们的模型，并且初始化变量：
sess = tf.Session()
sess.run(init)

# 训练模型1000次
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

saver = tf.train.Saver()
saver.save(sess, "ckpt/softmax.ckpt")
