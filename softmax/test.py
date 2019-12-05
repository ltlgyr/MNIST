# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def model():
	x = tf.placeholder("float",[None,784])
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	y_predict = tf.nn.softmax(tf.matmul(x,W) + b)
	return x,y_predict


x,y_predict=model()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, "ckpt/softmax.ckpt")

plt.figure()
for i in range(2):
	for j in range(10):
		im = Image.open('jpg/train_%d.jpg' %(i*10+j))
		im = im.convert('L')
		data = list(im.getdata())
		result = [x*1.0/255.0 for x in data]
		prediction=tf.argmax(y_predict,1)
		predint = prediction.eval(feed_dict={x: [result]}, session=sess)

		ax = plt.subplot(2,10, i*10+j +1)
		ax.set_title('%d' %int(predint[0]))
		plt.imshow(im)

plt.show()
