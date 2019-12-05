from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt


def imageprepare():
	im = Image.open('../jpg/train_0.jpg')
	plt.imshow(im)
	im = im.convert('L')
	data = list(im.getdata())
	#im = im.convert('L')
	#data = list(im.getdata())
	plt.imshow(im)
	result = [x*1.0/255.0 for x in data]
	return result

#定义初始化权重的函数
def weight_variavles(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))   
    return w

#定义一个初始化偏置的函数
def bias_variavles(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))   
    return b 


def model():
    tf.reset_default_graph()
    #1.建立数据的占位符 x [None, 784]  y_true [None, 10]
    with tf.variable_scope("date"):
        x = tf.placeholder(tf.float32, [None, 784])
        
        #y_true = tf.placeholder(tf.float32, [None, 10])
        
        
    
    #2.卷积层1  卷积:5*5*1,32个filter,strides= 1-激活-池化
    with tf.variable_scope("conv1"):
        #随机初始化权重
        w_conv1 = weight_variavles([5, 5, 1, 32])
        b_conv1 = bias_variavles([32])
        
        #对x进行形状的改变[None, 784] ----- [None,28,28,1]
        x_reshape = tf.reshape(x,[-1, 28, 28, 1])  #不能填None,不知道就填-1
        
        # [None,28, 28, 1] -------- [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding = "SAME") + b_conv1)
        
        #池化 2*2，步长为2，【None, 28,28, 32]--------[None,14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1],strides = [1,2,2,1],padding = "SAME")
        
    #3.卷积层2  卷积:5*5*32,64个filter,strides= 1-激活-池化
    with tf.variable_scope("conv2"):
        #随机初始化权重和偏置
        w_conv2 = weight_variavles([5, 5, 32, 64])
        b_conv2 = bias_variavles([64])
        
        #卷积、激活、池化
        #[None,14, 14, 32]----------【NOne, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2,strides=[1, 1, 1, 1], padding = "SAME") + b_conv2)
        
        #池化 2*2，步长为2 【None, 14,14，64]--------[None,7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1],strides = [1,2,2,1],padding = "SAME")
    
    #4.全连接层 [None,7, 7, 64] --------- [None, 7*7*64] * [7*7*64, 10]+[10] = [none, 10]
    with tf.variable_scope("fc"):
        #随机初始化权重和偏置:
        w_fc = weight_variavles([7 * 7 * 64, 1024])
        b_fc = bias_variavles([1024])
        
        #修改形状 [none, 7, 7, 64] ----------[None, 7*7*64]
        x_fc_reshape = tf.reshape(x_pool2,[-1,7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(x_fc_reshape, w_fc) + b_fc)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        w_fc1 = weight_variavles([1024, 10])
        b_fc1 = bias_variavles([10])
        
        #进行矩阵运算得出每个样本的10个结果[NONE, 10]
        #y_predict = tf.matmul(h_fc1_drop, w_fc1) + b_fc1
        y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc1) + b_fc1)
    return x, y_predict,keep_prob


def conv_fc():#获取数据
	result = imageprepare()

	#定义模型，得出输出
	x,y_predict,keep_prob = model()

	init_op = tf.global_variables_initializer()

	saver = tf.train.Saver()

	#开启会话运行
	#tf.reset_default_graph()
	sess = tf.Session() 
	sess.run(init_op)
	
	print(result)
	saver.restore(sess, "model/model.ckpt")
	prediction = tf.argmax(y_predict,1)
	predint = prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
	print(predint)
	print("recognize result: %d" %predint[0])

	return None

if __name__ == "__main__":
	conv_fc()
