from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import os
from keras import Model
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from thesis import attack
from art.classifiers import KerasClassifier

'''
def test_preds():

	model = load_model('models/vgg16_lin.h5')
	print(model.summary())
	
	correct_0 = 0
	incorrect_0 = 0
	correct_1 = 0
	incorrect_1 = 0

	sum = 0.0

	imgs = np.load("data/test_set.npy")
	ground_truth = np.load("data/test_labels.npy")

	for i in range(0, len(ground_truth)):
		pred = model.predict(imgs[i])
		pred = pred.astype(np.float64).flatten()
		print(pred)
		
		prediction = np.argmax(pred)
		sum += pred[prediction]
		if prediction == ground_truth[i]:
			if pred[prediction] > 1 - 1e-36:
				correct_0 += 1
			else:
				correct_1 += 1
		else:
			if pred[prediction] > 1 - 1e-36:
				incorrect_0 += 1
			else:
				incorrect_1 += 1


	print("Correct = 100%%: %d" % correct_0)
	print("Correct < 100%%: %d" % correct_1)
	print("Incorrect = 100%%: %d" % incorrect_0)
	print("Incorrect < 100%%: %d" % incorrect_1)
	

	print(sum/len(ground_truth))
'''

#training the model
def train_model(model):

	image_size = 224

	#flow from train directory
	train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
	
	#flow from validation directory 
	validation_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

	train_batchsize = 10
	val_batchsize = 5

	#load training images
	train_generator = train_datagen.flow_from_directory(
        "image_collection/raw_255/train",
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

	#load validation images
	validation_generator = validation_datagen.flow_from_directory(
        "image_collection/raw_255/valid",
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

	#compile model, specify learning rate of 0.0001
	model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])
		
	# Train the model
	history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=4,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)
	
	
	# Save the model
	model.save('models/vgg16_pretrained.h5')

	y_test = np.load('data/subsample_test_labels.npy')
	x_test = np.load('data/subsample_test_set.npy')
	x_test = x_test.squeeze()
	x_test = x_test.astype('float32')

	attacks = ["PGD"]
	model = KerasClassifier(model=model)
	for a in attacks:
		attack(model, x_test.squeeze(), y_test, a) 

	'''
	#summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig("VGG16_accuracy.png")
	'''

def train():
	
	#load in model
	vgg16 = VGG16(weights='imagenet', input_shape=(224,224,3))
	model = models.Sequential()
	#add all layers (except output layer) from vgg16 to model
	for i in range(0, len(vgg16.layers[:-1])):
		#if i == len(vgg16.layers[:-1]) - 1:
		#	model.add(layers.Dropout(0.2))
		model.add(vgg16.layers[i])
	#freeze all layers except last fully connected layers
	for layer in model.layers[:-4]:
		layer.trainable = False
	#model.add(layers.Dropout(0.2))
	#add output layer with 10 nodes instead of 1000
	model.add(layers.Dense(10, activation='softmax'))
	#call function to train model
	train_model(model)
	#test_preds(model)
	

if __name__ == "__main__":

	train()
	


