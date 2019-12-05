#j -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os 
from PIL import Image
import numpy as np

mnist = input_data.read_data_sets('./train', one_hot=True)
save_dir= 'jpg/'
if os.path.exists(save_dir) is False:
	os.makedirs(save_dir)

for i in range(20):
	image_array = mnist.train.images[i,:]
	image_array = image_array.reshape(28,28)
	image_array = image_array * 255
	filename = save_dir + 'train_%d.jpg' %i
	Image.fromarray(np.uint8(image_array)).convert('RGB').save(filename)


