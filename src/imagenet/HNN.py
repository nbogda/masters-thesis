from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, LeakyReLU
from keras.regularizers import l2
from keras.models import load_model
from keras import *
from keras.preprocessing import image
from keras.utils import np_utils
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Grad_VGG16:

	#builds the model
	def __init__(self, method):
		self.method = method
		model = Sequential()
		model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
		model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', ))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
		model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
		model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
		model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
		model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
		model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
		model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
		model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		# Add Fully Connected Layers
		model.add(Flatten())

		model.add(Dropout(0.2))
		model.add(Dense(512, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.13)))
		model.add(LeakyReLU(alpha=0))

		model.add(Dropout(0.2))
		model.add(Dense(512, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.13)))
		model.add(LeakyReLU(alpha=0))

		model.add(Dense(10, activation='softmax'))

		n_model = load_model("/data1/share/cinic-10/vgg16_10_class.h5")

		for i in range(len(model.layers[:-7])):
			model.layers[i].set_weights(n_model.layers[i].get_weights())
			
		del n_model

		for layer in model.layers[:-7]:
			layer.trainable = False

		op = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(loss = "categorical_crossentropy", optimizer = op, metrics=["accuracy"])
		self.model = model

	def train_model(self):

		image_size = 224
		#flow from train directory
		train_datagen = image.ImageDataGenerator()
		
		#flow from validation directory 
		validation_datagen = image.ImageDataGenerator()

		train_batchsize = 5
		val_batchsize = 10

		#load training images
		train_generator = train_datagen.flow_from_directory(
			"image_collection/%s/train" % self.method,
			target_size=(image_size, image_size),
			batch_size=train_batchsize,
			class_mode='categorical')
	 
		#load validation images
		validation_generator = validation_datagen.flow_from_directory(
			"image_collection/%s/valid" % self.method,
			target_size=(image_size, image_size),
			batch_size=val_batchsize,
			class_mode='categorical',
			shuffle=False)

		# Train the model
		self.history = self.model.fit_generator(
		  train_generator,
		  steps_per_epoch=train_generator.samples/train_generator.batch_size,
		  epochs=10,
		  validation_data=validation_generator,
		  validation_steps=validation_generator.samples/validation_generator.batch_size,
		  verbose=1)

		self.model.save('models/%s_vgg16_10_HNN.h5' % self.method)

	def plot_acc(self):

		fig = plt.figure()
		#summarize history for accuracy
		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
		plt.title('HNN %s Model Accuracy' % self.method)
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Valid'], loc='upper left')
		'''
		if not os.path.exists("results/%s" % self.method):
			os.mkdir("results/%s" % self.method)
		plt.savefig("results/%s/model_accuracy_HNN_%s_#2.png" % (self.method, self.method))
		'''
		plt.savefig("HNN_FOR_PAPER.png")

if __name__ == "__main__":

	methods = ["Med Gauss", "Resize"]

	for method in methods:
		print("TRAINING %s MODEL" % method)
		model = Grad_VGG16(method)
		model.train_model()
		#model.plot_acc()
		print("FINISHED %s MODEL" % method)
	
	'''
	method = 'raw_255'
	print("TRAINING %s MODEL" % method)
	model = Grad_VGG16(method)
	model.train_model()
	model.plot_acc()
	print("FINISHED %s MODEL" % method)
	'''	
