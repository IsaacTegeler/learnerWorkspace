from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback as c

K.set_image_dim_ordering('th')

'''
this pulls the loss for each batch into an array 
class LossHistory(c):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
'''


#-----------------pre building for size, shape, and defined constants------------------------------
img_width, img_height = 150, 150
count = 1
lossChange = 0.01
bestLossIndex = 0
batch_size = 100

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

#model.summary()
#compile network
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#---------------------grab training data----------------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/trainingData', 
						target_size=(img_width,img_height),
					 	batch_size=batch_size)
validation_generator = train_datagen.flow_from_directory('data/validation', 
						target_size=(img_width, img_height),
						batch_size=batch_size)
'''
history = LossHistory()

#this function controles how the learning rate is updated, taken care of with rmsprop so not needed
def scheduler(epoch):
	global count
	global bestLossIndex
	if epoch > 5:
		if (history.losses[bestLossIndex] - history.losses[epoch-1]) > lossChange:
			count += 1
			bestLossIndex = epoch
			print('updated learning rate, bestLossIndex: %d bestLoss: %lf' %(bestLossIndex, history.losses[bestLossIndex]))
	return (0.001/count)

change_lr = LearningRateScheduler(scheduler)
'''
#this trains the net
model.fit_generator(train_generator, steps_per_epoch = (12000/batch_size), epochs=25, validation_data=validation_generator, verbose=1, validation_steps=3)

model.save_weights('trained_net_2.h5')
