from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
K.set_image_dim_ordering('th')
#model prebuilding--------------------------------
img_width, img_height = 150, 150
batch_size = 100
classes = 100
model = Sequential()
model.add(ZeroPadding2D((1, 1) , batch_input_shape=(batch_size, 3, img_width, img_height)))
#model architecture------------------------------

#conLayers---------------------------------------
model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_3'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, activation = 'relu', name = 'con_4'))
model.add(MaxPooling2D((2,2)))
model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(64, 3, 3, activation = 'relu', name = 'con_5'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation = 'relu', name = 'con_6'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation = 'relu', name = 'con_7'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation = 'relu', name = 'con_8'))
model.add(MaxPooling2D((2,2)))
model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(128, 3, 3, activation = 'relu', name = 'con_9'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation = 'relu', name = 'con_10'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation = 'relu', name = 'con_11'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation = 'relu', name = 'con_12'))
model.add(MaxPooling2D((2,2)))
model.add(ZeroPadding2D((1, 1)))

model.add(Convolution2D(256, 3, 3, activation = 'relu', name = 'con_13'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation = 'relu', name = 'con_14'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation = 'relu', name = 'con_15'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation = 'relu', name = 'con_16'))
model.add(MaxPooling2D((2,2)))
model.add(ZeroPadding2D((1, 1)))

#fully_connected layers--------------------------
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(classes))

model.summary()
#------------------------------------------------
#post building-----------------------------------
#data pull---------------------------------------
#training----------------------------------------
