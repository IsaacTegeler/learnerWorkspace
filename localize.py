from keras import applications
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# build the VGG16 network
model = applications.VGG16(include_top=True,
                           weights='imagenet')

#prepare image for prediction
img_path = 'catTest.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


features = model.predict(x)
print('Predicted:', features)








'''
#regression head
reg_head = Seqential()
reg_head.add(Flatten(input_shape=model.output_shape[1:]))
reg_head.add(Dense(256))
reg_head.add(Dropout(0.25))
reg_head.add(Dense(256))
reg_head.add(Dropout(0.25))
reg_head.add(Activation('sigmoid'))
reg_head.add(Dense(4))

#classsification head
class_head = Sequential()
class_head.add(Flatten(input_shape=model.output_shape[1:]))
class_head.add(Dense(256))
class_head.add(Dropout(0.25))
class_head.add(Dense(256))
class_head.add(Dropout(0.25))
class_head.add(Activation('sigmoid'))
class_head.add(Dense(2))

#freeze the first 25 layers
for layer in model.layers[:25]:
	layer.trainable = False

#now what?

performe global average pooling at the end of the convolution layers
'''
