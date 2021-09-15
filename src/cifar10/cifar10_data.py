#this file is for opening the cifar10 data and putting it into a form that can be used in making a keras model
import pickle
import numpy as np
import os
import re
from keras.utils import np_utils

def unpickle_data(pwd):

	#the train and test data is organized in dictionary form like so: {img : label}
	train = {}
	test = {}

	x_train = [] #train images
	x_test = [] #test images
	y_train = [] #train labels
	y_test = [] #test labels

	for filename in os.listdir(pwd):
		#extract data from train images
		if re.match(r"data_batch_[0-9]", filename):
			with open("%s/%s" % (pwd, filename), 'rb') as fo:
				data = pickle.load(fo, encoding='bytes')

			#we havent initialized the dict yet, do so here
			if 'imgs' not in train:
				train['imgs'] = data[b'data']
				train['labels'] = np.array(data[b'labels'])
			#we have initialized the dict, concatenate the new data onto existing data
			else:
				train['imgs'] = np.concatenate((train['imgs'], data[b'data']))
				train['labels'] = np.concatenate((train['labels'], data[b'labels']))

		#same thing with test images
		elif re.match(r"test_batch", filename):
			with open("%s/%s" % (pwd, filename), 'rb') as fo:
				data = pickle.load(fo, encoding='bytes')

			#we havent initialized the dict yet, do so here
			if 'imgs' not in test:
				test['imgs'] = data[b'data']
				test['labels'] = np.array(data[b'labels'])
			#we have initialized the dict, concatenate the new data onto existing data
			else:
				test['imgs'] = np.concatenate((test['imgs'], data[b'data']))
				test['labels'] = np.concatenate((test['labels'], data[b'labels']))

	#now that we have all data loaded into dicts, reshape and put into arrays
	#since that is what keras wants
	for t in train['imgs']:
		x_train.append(np.transpose(np.reshape(t, (3, 32, 32)), (1, 2, 0)))
	for t in train['labels']:
		y_train.append(t)
	for t in test['imgs']:
		x_test.append(np.transpose(np.reshape(t, (3, 32, 32)), (1, 2, 0)))
	for t in test['labels']:
		y_test.append(t)

	#make into numpy arrays
	x_train = np.array(x_train)
	x_test = np.array(x_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)

	return x_train, y_train, x_test, y_test


if __name__ == "__main__":

	#open and read in the cifar-10 data into train and test sets
	x_train, y_train, x_test, y_test = unpickle_data("cifar-10-batches-py")

	#convert to float32
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	#normalize images to be between 0 and 1
	x_train /= 255
	x_test /= 255

	#one-hot encode the labels
	num_classes = 10
	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	#save the data we just extracted as numpy arrays to be used in making the keras model
	np.save('usable_data/x_train.npy', x_train)
	np.save('usable_data/y_train.npy', y_train)
	np.save('usable_data/x_test.npy', x_test)
	np.save('usable_data/y_test.npy', y_test)


