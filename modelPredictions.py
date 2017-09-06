from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from  keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import LearningRateScheduler
from keras.layers import GlobalAveragePooling2D

import numpy as np
K.set_image_dim_ordering('th')

#-----------------pre building for size, shape, and defined constants-------------------------------
img_width, img_height = 150, 150
count = 1
lossChange = 0.01
bestLossIndex = 0
batch_size = 1

img_path = 'testData/dogTest.jpeg'
img = load_img(img_path, target_size=(img_width, img_height))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

''' heat map stuff
a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
'''

model = Sequential()
model.add(ZeroPadding2D((1, 1) , batch_input_shape=(batch_size, 3, img_width, img_height)))
#---------------------Network Architecture--------------------------------------------------------

#Convolution layers -------------
model.add(Convolution2D(32, (3, 3), activation = 'relu', name = 'con_1'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, (3, 3), activation = 'relu', name = 'con_2'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))


model.add(Convolution2D(64, (3, 3), activation = 'relu', name = 'con_3'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation = 'relu', name = 'con_4'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1, 1)))

#Fully connected layers ---------
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'sigmoid', name='dense_3'))


#------------------look at the layers------------------------------------------------------------
layer_dict = dict([(layer.name, layer) for layer in model.layers])

import h5py

weights_path = 'trained_net_2.h5'

model.load_weights(weights_path, by_name=True)

print('Model loaded.')

#predict the class of the image
preds = model.predict(x, verbose=1)
print(preds.shape)
if(preds[0][0] == 1):
	print('cat')
else:
	print('dog')

