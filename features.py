from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Dense, Flatten
from keras import backend as K
K.set_image_dim_ordering('th')
img_width, img_height = 128, 128

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), batch_input_shape=(1, 3, img_width, img_height)))
first_layer = model.layers[-1]
# this is a placeholder tensor that will contain our generated images
input_img = first_layer.input

# build the rest of the network
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation = 'relu', name = 'dense_1'))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'relu', name = 'dense_2'))

model.summary()

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

import h5py

weights_path = 'vgg16_weights.h5'

f= h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
	if k >= len(model.layers):
	#dont need the last layers (the fully-connected ones) in the savefile
		break
	g = f['layer_{}'.format(k)]
	weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
	model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

#----------loss function--------------------------------------------------------------
layer_name = 'conv1_1'
filter_index = 0	#range is 0-511 as there is that many filters in the above layer

#this builds a loss function that maximizes the activation of the nth filter of the layer
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, filter_index, :, :])

#compute the gradient of the input picture
grads = K.gradients(loss, input_img)[0]

#normalization to normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

#this returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

#----------end------------------------------------------------------------------------------

import numpy as np

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

img = input_img_data[0]
img = deprocess_image(img)
imsave('/ConvLayers/%s_filter_%d.png' % (layer_name, filter_index), img)
