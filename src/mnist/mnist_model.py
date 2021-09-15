#figure out why this is giving zero gradients
from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import backend as k
import matplotlib.pylab as plt
import random

def add_noise(imgs):

	noisy_imgs = []

	for image in imgs:

		noise = image.reshape((28,28))

		#going through every row in image
		for i in range(0, len(noise)): 
			#choose a random index in the row to "noise"
			#how many indices to noise
			indices = 3
			index = random.randint(0,27)
			noise[i][index] =random.uniform(0.5, 1.0)
			already_used = [index]
			for j in range(indices - 1):
				index = None
				while 1:
					index = random.randint(0,27)
					if index not in already_used:
						break
				noise[i][index] = random.uniform(0.5, 1.0)

		noisy_imgs.append(noise)

	noisy_imgs = np.array(noisy_imgs, dtype='float32')
	return(noisy_imgs.reshape(len(imgs), 28, 28, 1)) 


batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#adding noise to images
#x_train = add_noise(x_train)
#x_test = add_noise(x_test)

#img = x_train[0].reshape((28,28))
#plt.imshow(img, cmap='gray')
#plt.show()

np.save('adv_imgs/y_test.npy', y_test)
np.save('adv_imgs/x_test.npy', x_test)

#build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
			 activation='relu',
			 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
		  optimizer=keras.optimizers.Adam(),
		  metrics=['accuracy'])

'''
class AccuracyHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.acc = []

	def on_epoch_end(self, batch, logs={}):
		self.acc.append(logs.get('acc'))
'''

# history = AccuracyHistory()

history = model.fit(x_train, y_train,
	  batch_size=batch_size,
	  epochs=epochs,
	  verbose=1,
	  validation_data=(x_test, y_test))
print(history)
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


fig = plt.figure()
#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("MNIST Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
# plt.show()
plt.savefig("mnist_model_accuracy_2.png")


model.save('model/mnist_model.h5')
