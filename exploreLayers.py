from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import numpy as np
K.set_image_dim_ordering('th')

'''
	To do today, change the imageGenerating function so that the image sizes match the size of the filters that they are being used for. This should be a parameter for the function.
'''

count = 0

#utility function to converta tensor to a valid image file
def deprocess_image(x):
	#normalize tensor: centor on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

#clip to [0,1]
	x += 0.5
	x = np.clip(x, 0, 1)
#convert to RGB array
	x *= 255
	x = x.transpose((1,2,0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x


#-----------------pre building for size, shape, and defined constants-------------------------------
img_width, img_height = 300, 300
count = 1
lossChange = 0.01
bestLossIndex = 0
batch_size = 1

model = Sequential()
model.add(ZeroPadding2D((1, 1) , batch_input_shape=(batch_size, 3, img_width, img_height)))
first_layer = model.layers[-1]
# this is a placeholder tensor that will contain our generated images
input_img = first_layer.input
input_img_data = np.random.random((1,3, img_width, img_height)) *20 + 128

#---------------------Network Architecture--------------------------------------------------------

#Convolution layers -------------
model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_1'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_2'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))



model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_4'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))


#------------------look at the layers------------------------------------------------------------
layer_dict = dict([(layer.name, layer) for layer in model.layers])

import h5py

weights_path = 'trained_net.h5'

model.load_weights(weights_path, by_name=True)

print('Model loaded.')

#----------loss function--------------------------------------------------------------
def generateImg(imageInput, layer_name, filter_index):
	global input_img_data
	#this builds a loss function that maximizes the activation of the nth filter of the layer
	layer_output = layer_dict[layer_name].output

	'''
	copyies everything from all points of the other 3 dimensions at filter_index. Where the
	following line of code has an array that is 4 dimensions and all values from the first 	  	  dimension when filter_index is the second dimesion, and all values in the 3rd and 4th 
	dimensions when filter_index is in the second dimension
	'''
	loss = K.mean(layer_output[:, filter_index, :, :]) 

	#compute the gradient of the input picture
	grads = K.gradients(loss, input_img)[0]

	#normalization to normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	#this returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	step = 1
	#start from a gray image with some noise
	input_img_data = np.random.random((1,3, img_width, img_height)) *20 + 128
	#run gradient ascent for 20 steps
	for i in range(20):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step
	
		print('Current loss value:', loss_value)
		if loss_value <= 0.:
		#skip the filters that get stuck at zero
			break


#--------------extract and display img----------------------------------------------------
from scipy.misc import imsave


#Do for both layers


for x in range(0 , 32):
	nm = generateImg(input_img, 'con_1', x)
	img = input_img_data[0]
	img = deprocess_image(img)
	imsave('./ConvLayers/%s_filter_%d.png' % ('con_1', x), img)
print("Con_1 completed")

for x in range(0 , 32):
	nm = generateImg(input_img, 'con_2', x)
	img = input_img_data[0]
	img = deprocess_image(img)
	imsave('./ConvLayers/%s_filter_%d.png' % ('con_2', x), img)
print("Con_2 completed")

for x in range(0 , 32):
	nm = generateImg(input_img, 'con_3', x)
	img = input_img_data[0]
	img = deprocess_image(img)
	imsave('./ConvLayers/%s_filter_%d.png' % ('con_3', x), img)
print("Con_3 complete")

for x in range(0 , 32):
	nm = generateImg(input_img, 'con_4', x)
	img = input_img_data[0]
	img = deprocess_image(img)
	imsave('./ConvLayers/%s_filter_%d.png' % ('con_4', x), img)
print("Con_4 complete")


